"""Masked-WT 20-AA profile and LLR from the repo-native ESM-2 650M (Task C).

WHAT THIS IS -- AND WHAT IT IS NOT
----------------------------------
This is a NEW score, computed from the same repo-native `esm2_t33_650M_UR50D`
checkpoint that produces the frozen representations, so the profile and the
representations are guaranteed to come from one model.

It is NOT `data/baseline_llr.csv`. That file was produced by the Hugging Face
`facebook/esm2_t30_150M_UR50D` model; its numbers are not comparable and are
never read, reused, or overwritten here. The one thing carried over from
`baseline_llr.py` is the *idea*: mask each position once and read every
substitution at that position off the single resulting distribution
(~376 forwards instead of ~4,500).

SCORING METHOD
--------------
`masked_marginal_wt`: for position p, the WT sequence is copied with residue p
replaced by <mask>, forwarded once, and the model's distribution at token index
p is taken as the profile for that position. The context is therefore always
wild-type -- the mutant residue never appears in the input.

    LLR = log P(MUT_aa | WT context, position p masked)
        - log P(WT_aa  | WT context, position p masked)

Logits and log_softmax are computed in FP32.

DEFINEDNESS
-----------
LLR is defined for missense substitutions only. Synonymous and in-frame indel
rows are emitted with `llr` missing and `llr_valid=False`; they are never given
a fabricated LLR of 0, which would be indistinguishable from a real neutral
score.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from .representation_cache import (
    BACKEND,
    MODEL_NAME,
    checkpoint_sha256,
    code_hash,
    file_sha256,
    git_commit,
    provenance_hash,
    sequence_hash,
)

#: Fixed ordering of the 20 standard amino acids. Stored alongside the profile so
#: a consumer never has to guess the column order.
AA20: List[str] = list("ACDEFGHIKLMNPQRSTVWY")
SCORING_METHOD = "masked_marginal_wt"
LLR_VERSION = "llr650m_v1"


@dataclass
class ProfileResult:
    """Profile plus the three separate cost numbers.

    `masked_positions` is how many WT positions were profiled, `masked_inputs`
    how many masked sequences were actually fed to the model (one per position
    here), and `model_forward_batch_calls` how many `model(tokens)` invocations
    that took -- roughly masked_inputs / batch_size. They are reported
    separately because a sequence count is not a forward count.
    """

    positions: np.ndarray          # [P] int, 1-based
    logprobs: np.ndarray           # [P, 20] float32, log-softmax over the full vocab
    wt_aa: List[str]               # [P]
    aa_order: List[str]
    masked_positions: int
    masked_inputs: int
    model_forward_batch_calls: int
    forward_seconds: float


@torch.no_grad()
def masked_wt_profile(
    encoder,
    wt_seq: str,
    positions: Sequence[int],
    *,
    batch_size: int = 8,
    progress_every: int = 0,
) -> ProfileResult:
    """One masked forward per position; returns the 20-AA log-prob profile.

    Every masked copy has the same length as the WT sequence, so a batch is
    always equal-length by construction.
    """
    model, alphabet = encoder.model, encoder.alphabet
    device = encoder.device
    positions = sorted({int(p) for p in positions})
    aa_idx = torch.tensor([alphabet.get_idx(a) for a in AA20], dtype=torch.long)

    rows: List[np.ndarray] = []
    t0 = time.perf_counter()
    n_masked_inputs = 0
    n_batch_calls = 0

    for start in range(0, len(positions), batch_size):
        chunk = positions[start : start + batch_size]
        data = [(f"pos_{p}", wt_seq) for p in chunk]
        _, _, tokens = alphabet.get_batch_converter()(data)
        tokens = tokens.to(device)
        # token layout is [BOS, res_1, ..., res_L, EOS] -> residue p is index p
        for b, p in enumerate(chunk):
            tokens[b, p] = alphabet.mask_idx

        logits = model(tokens, repr_layers=[])["logits"].float()
        n_batch_calls += 1
        for b, p in enumerate(chunk):
            logp = torch.log_softmax(logits[b, p], dim=-1)
            rows.append(logp.index_select(0, aa_idx.to(logp.device)).cpu().numpy())
        n_masked_inputs += len(chunk)
        if progress_every and n_masked_inputs % progress_every == 0:
            print(f"  masked {n_masked_inputs}/{len(positions)} positions", flush=True)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return ProfileResult(
        positions=np.asarray(positions, dtype=np.int32),
        logprobs=np.stack(rows).astype(np.float32),
        wt_aa=[wt_seq[p - 1] for p in positions],
        aa_order=list(AA20),
        masked_positions=len(positions),
        masked_inputs=n_masked_inputs,
        model_forward_batch_calls=n_batch_calls,
        forward_seconds=time.perf_counter() - t0,
    )


def llr_table(
    manifest_df,
    wt_seq: str,
    profile: ProfileResult,
) -> "Any":
    """Join the profile onto the rows. Label columns are never carried over.

    Missense rows get a real LLR; every other consequence gets llr=NaN and
    llr_valid=False, with `llr_undefined_reason` saying why.
    """
    import pandas as pd

    pos_row = {int(p): i for i, p in enumerate(profile.positions)}
    aa_col = {a: i for i, a in enumerate(profile.aa_order)}
    out: List[Dict[str, Any]] = []

    for _, raw in manifest_df.iterrows():
        row = raw.to_dict()
        cons = str(row.get("slim_consequence", "")).strip()
        pp = row.get("pp")
        rec: Dict[str, Any] = {
            "var_id": row.get("var_id"),
            "split": row.get("split"),
            "slim_consequence": cons,
            "pp": int(pp) if pp == pp and pp is not None else None,
            "wt_aa": None,
            "ref_aa": row.get("ref_aa"),
            "mut_aa": row.get("alt_aa"),
            "logp_wt": np.nan,
            "logp_mut": np.nan,
            "llr": np.nan,
            "llr_valid": False,
            "llr_undefined_reason": None,
            "scoring_method": SCORING_METHOD,
            "aa_order": "".join(profile.aa_order),
        }

        if cons != "missense":
            rec["llr_undefined_reason"] = f"llr_not_defined_for_{cons or 'unknown'}"
            out.append(rec)
            continue

        p = rec["pp"]
        if p is None or p not in pos_row:
            rec["llr_undefined_reason"] = "position_not_profiled"
            out.append(rec)
            continue

        wt_aa = wt_seq[p - 1]
        mut_aa = str(row.get("alt_aa") or "")
        rec["wt_aa"] = wt_aa
        rec["ref_matches_wt"] = str(row.get("ref_aa") or "") == wt_aa
        if wt_aa not in aa_col or mut_aa not in aa_col:
            rec["llr_undefined_reason"] = "non_standard_amino_acid"
            out.append(rec)
            continue

        lp = profile.logprobs[pos_row[p]]
        rec["logp_wt"] = float(lp[aa_col[wt_aa]])
        rec["logp_mut"] = float(lp[aa_col[mut_aa]])
        rec["llr"] = rec["logp_mut"] - rec["logp_wt"]
        rec["llr_valid"] = True
        out.append(rec)

    return pd.DataFrame(out)


def build_llr_provenance(
    *,
    manifest_path: Path,
    wt_seq: str,
    forward_precision: str,
    device: str,
    n_positions: int,
    masked_inputs: int,
    model_forward_batch_calls: int,
) -> Dict[str, Any]:
    import pandas as pd

    split_series = pd.read_csv(manifest_path, usecols=["var_id", "split"])
    prov = {
        "backend": BACKEND,
        "model_name": MODEL_NAME,
        "base_checkpoint_hash": checkpoint_sha256(),
        "adapter": "none",
        "adapter_state": "frozen",
        "task": "masked_wt_20aa_profile_and_llr",
        "llr_version": LLR_VERSION,
        "scoring_method": SCORING_METHOD,
        "aa_order": "".join(AA20),
        "repr_layers": [],
        "layer_convention": "lm_head logits (no hidden-state extraction)",
        "forward_precision": forward_precision,
        "logit_precision": "torch.float32",
        "log_softmax_precision": "torch.float32",
        "cache_precision": "float32",
        "device": device,
        "wt_sequence_hash": sequence_hash(wt_seq),
        "wt_length": len(wt_seq),
        "sequence_set_hash": sequence_hash(wt_seq),
        "masked_context": "wild_type_only",
        "n_profiled_positions": n_positions,
        "masked_positions": n_positions,
        "masked_inputs": masked_inputs,
        "model_forward_batch_calls": model_forward_batch_calls,
        "manifest_path": str(manifest_path),
        "manifest_hash": file_sha256(manifest_path),
        "split_schema_hash": hashlib.sha256(
            split_series.sort_values("var_id").to_csv(index=False).encode()
        ).hexdigest(),
        "alignment_version": "n/a_position_level_score",
        "window_rule": "n/a_single_masked_position",
        "code_hash": code_hash(
            ["src/embeddings/likelihood.py", "src/embeddings/esm_encoder.py"]
        ),
        "git": git_commit(),
        "contains_targets": False,
        "target_columns_excluded": ["z_score_D4_D14", "functional_classification"],
        "supersedes": None,
        "not_derived_from": "data/baseline_llr.csv (HF facebook/esm2_t30_150M_UR50D)",
    }
    prov["provenance_hash"] = provenance_hash(prov)
    return prov


def save_llr(
    out_dir: Path, table, profile: ProfileResult, provenance: Dict[str, Any]
) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "llr_650m.csv"
    npz_path = out_dir / "profile_650m.npz"
    prov_path = out_dir / "provenance.json"

    table.to_csv(csv_path, index=False)
    np.savez_compressed(
        npz_path,
        positions=profile.positions,
        logprobs=profile.logprobs,
        wt_aa=np.array(profile.wt_aa),
        aa_order=np.array(profile.aa_order),
        scoring_method=np.array(SCORING_METHOD),
        provenance_hash=np.array(provenance["provenance_hash"]),
    )
    prov_path.write_text(json.dumps(provenance, indent=2))
    return {
        "llr_csv": str(csv_path),
        "profile_npz": str(npz_path),
        "provenance_json": str(prov_path),
        "llr_csv_bytes": csv_path.stat().st_size,
        "profile_npz_bytes": npz_path.stat().st_size,
        "n_rows": int(len(table)),
        "n_llr_valid": int(table["llr_valid"].sum()),
    }


def load_llr(out_dir: Path, expected_provenance: Optional[Dict[str, Any]] = None):
    """Load the 650M LLR outputs, refusing a stale-provenance directory."""
    import pandas as pd

    from .representation_cache import StaleCacheError

    out_dir = Path(out_dir)
    prov = json.loads((out_dir / "provenance.json").read_text())
    if expected_provenance is not None:
        if provenance_hash(prov) != provenance_hash(expected_provenance):
            raise StaleCacheError(
                f"stale 650M LLR output at {out_dir}: provenance hash "
                f"{prov.get('provenance_hash')} != expected "
                f"{expected_provenance.get('provenance_hash')}"
            )
    return pd.read_csv(out_dir / "llr_650m.csv"), np.load(
        out_dir / "profile_650m.npz", allow_pickle=False
    ), prov
