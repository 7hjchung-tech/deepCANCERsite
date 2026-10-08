"""Section 12: inference-only structure-context sensitivity for E2 (best and last checkpoint).

1. normal input
2. neighbor SLOT order permuted (feature + metadata + mask moved together) -- the model has no
   positional encoding tied to the slot index itself (only to the offset/distance carried WITH
   each slot), so eval output must match the unpermuted run to numeric tolerance.
3. anchor kept, the other 8 slots' (feature, distance, offset) bundle replaced wholesale by
   another anchor's (donor's) own bundle -- donor id and the permutation seed are recorded.
   This is reported purely as input-perturbation sensitivity, not a causal/realism claim (the
   donor's neighborhood pasted onto a different anchor is not a biologically meaningful structure).

If E2's best epoch is 0, delta is identically 0 there, so ALL three probes trivially agree --
expected, not re-derived as a new finding.

Output -> analysis/stage2_neighborhood/sensitivity_{best,last}.csv + summary json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.stage1.checkpoint import default_reference  # noqa: E402
from src.stage1.engine import group_of  # noqa: E402
from src.stage2.diagnostics import DEFAULT_STAGE1, load_setup  # noqa: E402
from src.stage2.neighbor_engine import _summarise, forward_batch, make_loader  # noqa: E402
from src.stage2.neighbor_model import NeighborhoodFiLMModel  # noqa: E402
from src.stage2.neighbor_structure import ResidueStructureTokenizer, WTNeighborStore  # noqa: E402
from src.stage2.stage1_adapter import load_stage1_handle  # noqa: E402

OUT = ROOT / "analysis/stage2_neighborhood"
IN_DIR = ROOT / "runs/stage2_neighborhood/E2/seed42"
CACHE = ROOT / "data/structure/results/wt_neighbor_cache.npz"


def load_model(which: str, st, handle):
    ck = torch.load(IN_DIR / f"{which}.pt", map_location="cuda", weights_only=False)
    tok = ResidueStructureTokenizer(d_s=32, n_bins=4)
    store = WTNeighborStore(CACHE)
    train_anchors = [int(e["row"]["pp"]) for e in st.by_split["train"]]
    tok.fit_preprocessing(store.raw_for_positions(store.fit_positions_for_train_anchors(train_anchors)))
    tok.load_state_dict(ck["tokenizer"])
    model = NeighborhoodFiLMModel(d=ck["d_model"])
    model.load_state_dict(ck["model"])
    return model.to("cuda").eval(), tok.to("cuda").eval(), ck["epoch"]


@torch.no_grad()
def run_probe(model, tok, handle, loader, perturb: str | None, rng: np.random.Generator) -> dict:
    """perturb: None (unperturbed) or "permute" (neighbor slots 1..8 permuted together,
    anchor slot 0 fixed). Donor-swap is a separate function (run_donor_swap) since it needs
    the WTNeighborStore to pull another anchor's own bundle, not just a permutation."""
    preds, y, groups, var_ids, deltas = [], [], [], [], []
    for batch in loader:
        raw = dict(batch["struct_raw"])
        B = raw["valid"].shape[0]
        if perturb == "permute":
            for b in range(B):
                perm = rng.permutation(8) + 1          # permute slots 1..8, keep slot 0 (anchor) fixed
                order = np.concatenate([[0], perm])
                for k in ("continuous", "ss", "distance", "offset", "is_anchor", "valid"):
                    raw[k][b] = raw[k][b][order]
        batch = dict(batch)
        batch["struct_raw"] = raw
        out, s1 = forward_batch(model, tok, handle, batch, "cuda")
        preds.append(out["pred"].cpu().numpy()); y.append(batch["label"].numpy())
        groups += [group_of(t) for t in batch["edit_type"]]
        var_ids += batch["var_id"]; deltas.append(out["delta"].cpu().numpy())
    return {"pred": np.concatenate(preds), "label": np.concatenate(y), "group": np.array(groups),
           "var_id": var_ids, "delta": np.concatenate(deltas)}


@torch.no_grad()
def run_donor_swap(model, tok, handle, st, store, mode_entries, rng: np.random.Generator) -> dict:
    """Rebuilds val batches with the 8 non-anchor slots replaced by a DONOR anchor's own bundle
    (feature+metadata moved together, donor's own values, not recomputed relative to our anchor)."""
    from src.stage2.neighbor_engine import make_collate
    var_id_to_pp = {e["var_id"]: int(e["row"]["pp"]) for e in mode_entries}
    all_positions = store.positions.tolist()
    donors = {}
    preds, y, groups, var_ids, deltas, donor_ids = [], [], [], [], [], []
    from src.stage1.dataset import Stage1Dataset
    ds = Stage1Dataset(mode_entries, st.cache, st.window, st.layers, meta_scaler=handle.meta_scaler)
    collate = make_collate(store, "e2", var_id_to_pp)
    loader = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=False, collate_fn=collate)
    for batch in loader:
        raw = dict(batch["struct_raw"])
        B = raw["valid"].shape[0]
        donor_anchor = rng.choice(all_positions, size=B).tolist()
        donor_raw = store.raw_for_anchors(donor_anchor, "e2")
        for k in ("continuous", "ss", "distance", "offset", "is_anchor"):
            raw[k] = raw[k].clone()
            raw[k][:, 1:] = donor_raw[k][:, 1:]        # keep slot 0 (our own anchor); replace slots 1..8
        batch = dict(batch); batch["struct_raw"] = raw
        out, s1 = forward_batch(model, tok, handle, batch, "cuda")
        preds.append(out["pred"].cpu().numpy()); y.append(batch["label"].numpy())
        groups += [group_of(t) for t in batch["edit_type"]]
        var_ids += batch["var_id"]; deltas.append(out["delta"].cpu().numpy()); donor_ids += donor_anchor
    return {"pred": np.concatenate(preds), "label": np.concatenate(y), "group": np.array(groups),
           "var_id": var_ids, "delta": np.concatenate(deltas), "donor_pos": donor_ids}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    st = load_setup(DEFAULT_STAGE1, "cuda", splits=("train", "val"))
    ref = default_reference(st.cfg.get("esm_checkpoint", "esm2_t33_650M_UR50D"), st.cache.wt_hash)
    handle = load_stage1_handle(DEFAULT_STAGE1, ref, "cuda")
    store = WTNeighborStore(CACHE)

    summary = {}
    for which in ("best", "last"):
        model, tok, epoch = load_model(which, st, handle)
        loader = make_loader(st.by_split["val"], st.cache, st.window, st.layers, handle, store, "e2", 128, shuffle=False)
        normal = run_probe(model, tok, handle, loader, None, np.random.default_rng(0))
        permuted = run_probe(model, tok, handle, loader, "permute", np.random.default_rng(0))
        rng_donor = np.random.default_rng(1)
        donor = run_donor_swap(model, tok, handle, st, store, st.by_split["val"], rng_donor)

        assert normal["var_id"] == permuted["var_id"] == donor["var_id"]
        perm_diff = np.abs(normal["pred"] - permuted["pred"])
        donor_diff = np.abs(normal["pred"] - donor["pred"])
        m_normal = _summarise(normal["label"], normal["pred"], normal["group"])
        m_perm = _summarise(permuted["label"], permuted["pred"], permuted["group"])
        m_donor = _summarise(donor["label"], donor["pred"], donor["group"])

        pd.DataFrame({"var_id": normal["var_id"], "label": normal["label"], "pred_normal": normal["pred"],
                     "pred_permuted": permuted["pred"], "pred_donor_swap": donor["pred"],
                     "donor_pos": donor["donor_pos"], "abs_diff_permuted": perm_diff,
                     "abs_diff_donor_swap": donor_diff, "delta_normal": normal["delta"]}).to_csv(
            OUT / f"sensitivity_{which}.csv", index=False)

        summary[which] = {
            "epoch": epoch, "delta_abs_mean_normal": float(np.abs(normal["delta"]).mean()),
            "val_subset_normal": m_normal["subset"], "val_subset_permuted": m_perm["subset"],
            "val_subset_donor_swap": m_donor["subset"],
            "permutation_invariance_max_abs_diff": float(perm_diff.max()), "permutation_invariance_mean_abs_diff": float(perm_diff.mean()),
            "donor_swap_mean_abs_pred_change": float(donor_diff.mean()), "donor_swap_max_abs_pred_change": float(donor_diff.max()),
        }
        print(which, json.dumps(summary[which], indent=2))
    (OUT / "sensitivity_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[sensitivity] wrote {OUT}")


if __name__ == "__main__":
    main()
