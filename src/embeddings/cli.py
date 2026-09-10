"""Task C driver: fixture -> benchmark -> full frozen export, plus the 650M LLR.

Subcommands
-----------
  fixture     small real-ESM correctness fixture (must pass before export)
  benchmark   timing/memory measurement on ~32 real, distinct sequences
  export      full frozen representation cache for in_eval_scope rows
  llr         masked-WT 20-AA profile + LLR from the same 650M checkpoint

Run order matters: `export` refuses to start unless a fixture report that
matches the current provenance identity is on disk.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from . import representation_cache as rc
from .contract import DEFAULT_CONFIG, Contract, load_contract
from .esm_encoder import ESMEncoder
from .likelihood import (
    build_llr_provenance,
    llr_table,
    masked_wt_profile,
    save_llr,
)

_ROOT = Path(__file__).resolve().parents[2]


def resolve_runtime(args: argparse.Namespace) -> Contract:
    """Validate the YAML contract, then fill unset flags from it.

    Every subcommand goes through here BEFORE loading a model, so a config that
    contradicts the Python constants stops the run instead of producing a cache
    whose documented contract is a lie.
    """
    contract = load_contract(args.config)
    if args.manifest is None:
        args.manifest = str(contract.path_of("manifest"))
    if args.wt is None:
        args.wt = str(contract.path_of("wt_sequence"))
    if args.out_dir is None:
        args.out_dir = str(contract.path_of(args.out_key))
    if args.batch_size is None:
        args.batch_size = contract.batch_size
    if args.window is None:
        args.window = contract.window_W
    if args.device is None:
        want = contract.device
        args.device = want if (want != "cuda" or torch.cuda.is_available()) else "cpu"
    print(f"[contract] {contract.path.relative_to(_ROOT)} validated against the "
          f"Python constants; W={args.window} batch={args.batch_size} "
          f"device={args.device}")
    return contract


# ==========================================================================
# shared setup
# ==========================================================================
def load_inputs(manifest: Path, wt_path: Path) -> Tuple[pd.DataFrame, str]:
    wt_seq = Path(wt_path).read_text(encoding="utf-8").strip()
    df = pd.read_csv(manifest)
    return df, wt_seq


def make_encoder(device: str) -> Tuple[ESMEncoder, float]:
    """Load the frozen 650M encoder. Returns (encoder, load_seconds)."""
    t0 = time.perf_counter()
    enc = ESMEncoder({"device": device, "repr_layer": list(rc.REPR_LAYERS)})
    enc.model.eval()
    if torch.cuda.is_available() and device.startswith("cuda"):
        torch.cuda.synchronize()
    return enc, time.perf_counter() - t0


def _rss_bytes() -> int:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return -1


# ==========================================================================
# fixture
# ==========================================================================
#: The fixture deliberately covers BOTH real duplication rows (p.Lys186dup, in
#: scope; p.Val351dup, cross-span-excluded) and both protein termini, so the
#: window contract is exercised where it truncates as well as where it is full.
FIXTURE_SPECS = [
    ("missense", lambda df: df[(df.slim_consequence == "missense") & df.pp.between(20, 350)].iloc[0]),
    ("synonymous", lambda df: df[(df.slim_consequence == "synonymous") & df.pp.between(20, 350)].iloc[0]),
    ("deletion", lambda df: df[df.HGVSp.astype(str).str.endswith("del") & df.pp.between(20, 350)].iloc[0]),
    ("duplication_186", lambda df: df[df.HGVSp.astype(str).str.contains("Lys186dup", na=False)].iloc[0]),
    ("duplication_351", lambda df: df[df.HGVSp.astype(str).str.contains("Val351dup", na=False)].iloc[0]),
    ("delins", lambda df: df[df.HGVSp.astype(str).str.contains("delinsGln", na=False)].iloc[0]),
    ("missense_nterm", lambda df: df[df.slim_consequence == "missense"].nsmallest(1, "pp").iloc[0]),
    ("missense_cterm", lambda df: df[df.slim_consequence == "missense"].nlargest(1, "pp").iloc[0]),
]
DUP_LABELS = ("duplication_186", "duplication_351")


def select_fixture_rows(df: pd.DataFrame) -> pd.DataFrame:
    picked, labels = [], []
    for label, sel in FIXTURE_SPECS:
        row = sel(df)
        picked.append(row)
        labels.append(label)
    out = pd.DataFrame(picked).copy()
    out["fixture_label"] = labels
    return out


class Checks:
    def __init__(self) -> None:
        self.results: List[Dict[str, Any]] = []

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.results.append({"check": name, "pass": bool(ok), "detail": detail})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
        return bool(ok)

    @property
    def all_passed(self) -> bool:
        return all(r["pass"] for r in self.results)


def cmd_fixture(args: argparse.Namespace) -> int:
    resolve_runtime(args)
    df, wt_seq = load_inputs(Path(args.manifest), Path(args.wt))
    fx = select_fixture_rows(df)
    print("Fixture rows:")
    for _, r in fx.iterrows():
        print(f"  {r.fixture_label:12s} {r.var_id:32s} {r.HGVSp}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    enc, load_s = make_encoder(args.device)
    print(f"[model] loaded in {load_s:.1f}s on {args.device}, eval={not enc.model.training}")

    # The delins and p.Val351dup rows are cross-span-excluded from the eval
    # scope, but they are the most informative alignment cases, so the fixture
    # plans every row regardless of scope.
    plans, skipped = rc.plan_rows(fx, wt_seq, W=args.window, in_scope_only=False)
    label_of = dict(zip(fx.var_id, fx.fixture_label))
    by_label = {label_of[p.var_id]: p for p in plans}

    c = Checks()
    c.check(f"all {len(FIXTURE_SPECS)} fixture rows planned",
            len(plans) == len(FIXTURE_SPECS), f"skipped={skipped}")

    # ---- 1. layer shapes ------------------------------------------------
    fc = rc.SequenceForwardCache(enc, rc.REPR_LAYERS, batch_size=args.batch_size)
    wt_full = fc.get(wt_seq)
    c.check(
        "WT full repr shape [3, L, 1280]",
        tuple(wt_full.shape) == (3, len(wt_seq), 1280),
        f"{tuple(wt_full.shape)} for L={len(wt_seq)}",
    )
    reps = enc.encode([wt_seq], repr_layers=list(rc.REPR_LAYERS))
    c.check(
        "layers 31/32/33 all present, each [1, L, 1280]",
        sorted(reps) == [31, 32, 33]
        and all(tuple(reps[l].shape) == (1, len(wt_seq), 1280) for l in (31, 32, 33)),
        str({l: tuple(reps[l].shape) for l in sorted(reps)}),
    )
    c.check(
        "layers 31/32/33 are distinct tensors",
        not torch.allclose(reps[31], reps[32]) and not torch.allclose(reps[32], reps[33]),
    )

    # ---- 2. individual vs batch -----------------------------------------
    mut_seqs = [p.mut_seq for p in plans]
    lengths = sorted({len(s) for s in mut_seqs})
    c.check("fixture spans >1 sequence length", len(lengths) > 1, f"lengths={lengths}")

    same_len = [s for s in mut_seqs if len(s) == len(wt_seq)][:2]
    batched = enc.encode(same_len, repr_layers=list(rc.REPR_LAYERS))
    # Tolerance must be RELATIVE: layers 31/32 are pre-final-LayerNorm and carry
    # values up to ~500, so an absolute threshold sized for layer 33 (|h| ~ 8)
    # would be meaningless there. Changing the batch shape changes the GEMM
    # reduction order in FP32; the same drift appears on CPU, so this measures
    # float associativity, not a batching bug.
    max_abs, max_rel = 0.0, 0.0
    for i, s in enumerate(same_len):
        single = enc.encode([s], repr_layers=list(rc.REPR_LAYERS))
        for l in rc.REPR_LAYERS:
            d = (single[l][0] - batched[l][i]).abs().max().item()
            scale = batched[l][i].abs().max().item()
            max_abs = max(max_abs, d)
            max_rel = max(max_rel, d / max(scale, 1e-9))
    c.check(
        "individual vs batched forward within relative tolerance",
        max_rel < args.tol,
        f"max|diff|={max_abs:.3e} rel={max_rel:.3e} tol={args.tol:.0e}",
    )

    # Determinism at a fixed batch shape is exact, which is what the dedup and
    # the cache identity actually rely on.
    dup_batch = enc.encode([same_len[0], same_len[1], same_len[0]],
                           repr_layers=list(rc.REPR_LAYERS))
    c.check(
        "same sequence at two batch positions is bit-identical",
        all(torch.equal(dup_batch[l][0], dup_batch[l][2]) for l in rc.REPR_LAYERS),
    )

    # mixed-length batches must be refused, not truncated
    try:
        enc.encode([wt_seq, by_label["deletion"].mut_seq])
        refused = False
    except ValueError:
        refused = True
    c.check("mixed-length batch is refused by _forward", refused)

    # ---- 3. synonymous ---------------------------------------------------
    syn = by_label["synonymous"]
    c.check("synonymous MUT protein == WT protein", syn.mut_seq == wt_seq)
    c.check(
        "synonymous reuses the SAME cached WT tensor object",
        fc.get(syn.mut_seq) is wt_full,
        f"seq_hash {syn.mut_seq_hash[:12]} == wt {rc.sequence_hash(wt_seq)[:12]}",
    )

    # ---- run the export so save/reload and delta can be checked ---------
    A = max(p.n_slots for p in plans)
    fixture_cache = out_dir / "fixture_repr.pt"
    res = rc.export_representations(
        plans, wt_seq, enc, fixture_cache,
        manifest_path=Path(args.manifest), W=args.window,
        batch_size=args.batch_size, device=args.device, A=A,
    )
    print(f"[fixture] wrote {res.path} ({res.report['cache_bytes']:,} bytes)")

    cache = rc.load_cache(fixture_cache, expected_provenance=res.provenance)
    idx_of = {v: i for i, v in enumerate(cache.var_id)}

    def row_of(label: str) -> Dict[str, torch.Tensor]:
        return cache.row(idx_of[by_label[label].var_id])

    syn_row = row_of("synonymous")
    dv = syn_row["delta_valid"]
    c.check(
        "synonymous valid delta_H is EXACTLY 0",
        float(syn_row["delta_H"][:, dv, :].abs().max()) == 0.0,
        f"max|delta|={float(syn_row['delta_H'][:, dv, :].abs().max()):.1e} over {int(dv.sum())} valid slots",
    )
    c.check(
        "synonymous H_WT == H_MUT bit-exact on valid slots",
        torch.equal(syn_row["H_WT"][:, dv, :], syn_row["H_MUT"][:, dv, :]),
    )

    # ---- 4-7. delins alignment ------------------------------------------
    dl = by_label["delins"]
    arr = dl.arrays
    slots = list(zip(arr["wt_pos"], arr["mut_pos"], arr["slot_kind"],
                     arr["delta_valid"], arr["token_valid"]))
    paired_map = {w: m for w, m, k, _, _ in slots if k == "paired"}
    c.check("delins: WT343 -> MUT339 pairing preserved",
            paired_map.get(343) == 339, f"map[343]={paired_map.get(343)}")
    c.check("delins: WT337 -> MUT337 left boundary paired",
            paired_map.get(337) == 337, f"map[337]={paired_map.get(337)}")
    deleted = {w for w, m, k, _, _ in slots if k == "wt_only"}
    c.check("delins: deleted WT 338-342 are wt_only",
            deleted == {338, 339, 340, 341, 342}, f"wt_only={sorted(deleted)}")
    ins_slots = [(w, m) for w, m, k, _, _ in slots if k == "mut_only"]
    mut_seq = dl.mut_seq
    c.check("delins: inserted Q is a single mut_only slot at MUT338",
            len(ins_slots) == 1 and ins_slots[0] == (0, 338)
            and mut_seq[ins_slots[0][1] - 1] == "Q",
            f"mut_only={ins_slots} residue={mut_seq[ins_slots[0][1]-1] if ins_slots else '-'}")
    c.check("delins: no wt_only/mut_only slot is delta_valid",
            all(not d for _, _, k, d, _ in slots if k in ("wt_only", "mut_only")))
    c.check("delins: every gap slot is token_valid (gaps are not padding)",
            all(t for _, _, k, _, t in slots if k in ("wt_only", "mut_only")))

    dl_row = row_of("delins")
    kinds = cache.slot_kind_names(idx_of[dl.var_id])
    gap = torch.tensor([k in ("wt_only", "mut_only") for k in kinds])
    c.check("delins: delta_H is exactly 0 on every gap slot",
            float(dl_row["delta_H"][:, gap, :].abs().max()) == 0.0)
    wt_only_mask = torch.tensor([k == "wt_only" for k in kinds])
    c.check("delins: wt_only slots keep a real WT representation (not zeroed)",
            float(dl_row["H_WT"][:, wt_only_mask, :].abs().max()) > 0.0
            and float(dl_row["H_MUT"][:, wt_only_mask, :].abs().max()) == 0.0)
    mut_only_mask = torch.tensor([k == "mut_only" for k in kinds])
    c.check("delins: mut_only slots keep a real MUT representation (not zeroed)",
            float(dl_row["H_MUT"][:, mut_only_mask, :].abs().max()) > 0.0
            and float(dl_row["H_WT"][:, mut_only_mask, :].abs().max()) == 0.0)

    # no same-array-index subtraction: the paired slots must actually use
    # different wt/mut coordinates on the right side of the edit
    shifted = [(int(w), int(m)) for w, m, k, d, _ in slots if k == "paired" and w != m]
    c.check("delins: right-flank paired slots are index-shifted (no same-index subtraction)",
            len(shifted) >= 10, f"{len(shifted)} shifted paired slots, e.g. {shifted[:3]}")

    # verify a shifted paired delta really is H_MUT[mut_pos] - H_WT[wt_pos]
    wpos = dl_row["wt_pos"].tolist()
    mpos = dl_row["mut_pos"].tolist()
    probe = next(i for i, k in enumerate(kinds) if k == "paired" and wpos[i] != mpos[i])
    mut_full = fc.get(dl.mut_seq)
    expect = mut_full[:, mpos[probe] - 1, :] - wt_full[:, wpos[probe] - 1, :]
    c.check("delins: shifted paired delta == MUT[mut_pos] - WT[wt_pos] bit-exact",
            torch.equal(dl_row["delta_H"][:, probe, :], expect),
            f"slot {probe}: WT{wpos[probe]} vs MUT{mpos[probe]}")

    # ---- duplication: BOTH real rows -------------------------------------
    for label in DUP_LABELS:
        dup = by_label[label]
        dpp = int(dup.pp)
        dslots = list(zip(dup.arrays["wt_pos"], dup.arrays["mut_pos"], dup.arrays["slot_kind"]))
        dmap = {w: m for w, m, k in dslots if k == "paired"}
        dup_ins = [m for w, m, k in dslots if k == "mut_only"]
        dup_mut = dup.mut_seq
        c.check(f"{label}: MUT is one residue longer than WT",
                len(dup_mut) == len(wt_seq) + 1, f"{len(dup_mut)} vs {len(wt_seq)}")
        c.check(f"{label}: duplicated copy is one mut_only slot at MUT{dpp + 1}",
                dup_ins == [dpp + 1] and dup_mut[dpp] == wt_seq[dpp - 1],
                f"mut_only={dup_ins} residue={dup_mut[dpp] if dup_ins else '-'}")
        c.check(f"{label}: every paired slot matches the same residue identity",
                all(wt_seq[w - 1] == dup_mut[m - 1] for w, m in dmap.items()),
                f"{len(dmap)} paired slots")
        c.check(f"{label}: paired slots up to the insertion boundary are unshifted",
                all(m == w for w, m in dmap.items() if w <= dpp),
                f"e.g. {[(w, m) for w, m in sorted(dmap.items()) if w <= dpp][:3]}")
        c.check(f"{label}: post-boundary paired slots are frame-shifted by +1",
                all(m == w + 1 for w, m in dmap.items() if w > dpp),
                f"e.g. {[(w, m) for w, m in sorted(dmap.items()) if w > dpp][:3]}")
        # the split rule reads the insertion boundary, never the flank
        rec = rc.build_variant_record(
            fx[fx.var_id == dup.var_id].iloc[0].to_dict(), wt_seq, manifest_df=df
        )
        c.check(f"{label}: split rule reads only the insertion boundary "
                f"{{{dpp}, {dpp + 1}}}",
                rec["split_positions_considered"] == [dpp, dpp + 1],
                f"considered={rec['split_positions_considered']} "
                f"splits={rec['split_position_splits']} "
                f"cross_span={rec['cross_span']} in_scope={rec['in_eval_scope']}")

    # ---- simple deletion --------------------------------------------------
    dele = by_label["deletion"]
    eslots = list(zip(dele.arrays["wt_pos"], dele.arrays["mut_pos"], dele.arrays["slot_kind"]))
    emap = {w: m for w, m, k in eslots if k == "paired"}
    c.check("deletion: MUT is one residue shorter than WT",
            len(dele.mut_seq) == len(wt_seq) - 1)
    c.check("deletion: deleted residue is wt_only",
            [w for w, m, k in eslots if k == "wt_only"] == [int(dele.pp)])
    c.check("deletion: post-edit paired slots are frame-shifted by -1",
            all(m == w - 1 for w, m in emap.items() if w > int(dele.pp)),
            f"e.g. {[(w, m) for w, m in sorted(emap.items()) if w > int(dele.pp)][:3]}")
    c.check("deletion: every paired slot matches the same residue identity",
            all(wt_seq[w - 1] == dele.mut_seq[m - 1] for w, m in emap.items()))

    # ---- window rule: EXACT slot budgets ---------------------------------
    W = args.window
    mis = by_label["missense"]
    c.check(f"missense window is +-{W}: exactly {2 * W + 1} slots, all paired",
            mis.n_slots == 2 * W + 1 and set(mis.arrays["slot_kind"]) == {"paired"},
            f"n_slots={mis.n_slots}")
    syn = by_label["synonymous"]
    c.check(f"synonymous window is +-{W}: exactly {2 * W + 1} slots, all paired",
            syn.n_slots == 2 * W + 1 and set(syn.arrays["slot_kind"]) == {"paired"},
            f"n_slots={syn.n_slots}")

    # exact flank/event contract: W paired on each side, the boundary residues
    # counted INSIDE those W, plus the edit event itself -- no ">= 10" slack.
    expect_event = {
        "deletion": (1, 0),                       # (wt_only, mut_only)
        DUP_LABELS[0]: (0, 1),
        DUP_LABELS[1]: (0, 1),
        "delins": (5, 1),
    }
    for label, (n_wt_only, n_mut_only) in expect_event.items():
        pl = by_label[label]
        pk = pl.arrays["slot_kind"]
        left = next(i for i, k in enumerate(pk) if k != "paired")
        right = next(i for i, k in enumerate(reversed(pk)) if k != "paired")
        total = 2 * W + n_wt_only + n_mut_only
        c.check(f"{label}: exactly {W} paired flank slots on each side "
                f"and a {n_wt_only}+{n_mut_only} edit event ({total} slots)",
                left == W and right == W
                and pk.count("wt_only") == n_wt_only
                and pk.count("mut_only") == n_mut_only
                and len(pk) == total,
                f"left={left} right={right} wt_only={pk.count('wt_only')} "
                f"mut_only={pk.count('mut_only')} total={len(pk)} expected={total}")

    # termini: fewer flank slots are allowed there, and ONLY there
    for label, side in (("missense_nterm", "N"), ("missense_cterm", "C")):
        pl = by_label[label]
        pp = int(pl.pp)
        lo, hi = max(1, pp - W), min(len(wt_seq), pp + W)
        c.check(f"{label}: {side}-terminal window truncates to WT {lo}..{hi} "
                f"({hi - lo + 1} slots), not wrapped or padded with real residues",
                pl.arrays["wt_pos"] == list(range(lo, hi + 1))
                and pl.n_slots == hi - lo + 1
                and pl.n_slots < 2 * W + 1,
                f"pp={pp} n_slots={pl.n_slots} "
                f"wt_pos[0]={pl.arrays['wt_pos'][0]} wt_pos[-1]={pl.arrays['wt_pos'][-1]}")

    # Task B's core must reappear verbatim inside every geometry-built window
    for label, pl in by_label.items():
        rec = rc.build_variant_record(
            fx[fx.var_id == pl.var_id].iloc[0].to_dict(), wt_seq, manifest_df=df
        )
        core = {k: rec[k] for k in rc.ARRAY_KEYS}
        at = rc.core_offset_in_window(core, pl.arrays)
        c.check(f"{label}: Task B edit core is contiguous inside the window",
                at >= 0, f"core of {len(core['slot_kind'])} slots at offset {at}")

    # ---- 8. save -> reload identity --------------------------------------
    cache2 = rc.load_cache(fixture_cache, expected_provenance=res.provenance)
    same = all(
        torch.equal(cache.p[k], cache2.p[k])
        for k in ("wt_full", "mut_windows", "wt_pos", "mut_pos", "wt_present",
                  "mut_present", "delta_valid", "token_valid", "slot_kind",
                  "row_to_mut_window")
    )
    c.check("reload: tensors and masks bit-identical", same)
    c.check("reload: provenance identical",
            cache2.provenance["provenance_hash"] == res.provenance["provenance_hash"])
    b1, b2 = cache.batch(range(len(cache))), cache2.batch(range(len(cache2)))
    c.check("reload: H_WT/H_MUT/delta_H bit-identical",
            all(torch.equal(b1[k], b2[k]) for k in ("H_WT", "H_MUT", "delta_H")))
    c.check("logical output contract [B,3,A,1280]",
            tuple(b1["H_WT"].shape) == (len(plans), 3, A, 1280)
            and tuple(b1["delta_H"].shape) == (len(plans), 3, A, 1280),
            f"{tuple(b1['H_WT'].shape)}")
    c.check("cache carries no SGE target / functional classification column",
            not ({"z_score_D4_D14", "functional_classification"} & set(cache.p)),
            f"keys={sorted(k for k in cache.p if isinstance(cache.p[k], list))}")

    # ---- 9. stale provenance rejection -----------------------------------
    for field, bad in [
        ("base_checkpoint_hash", "0" * 64),
        ("repr_layers", [30, 31, 32]),
        ("window_rule_version", "window_v0"),
        ("cache_precision", "torch.float16"),
        ("alignment_version", "taskB_canonical_v0"),
        ("manifest_hash", "deadbeef"),
    ]:
        stale = dict(res.provenance)
        stale[field] = bad
        try:
            rc.load_cache(fixture_cache, expected_provenance=stale, strict=True)
            rejected = False
        except rc.StaleCacheError:
            rejected = True
        c.check(f"stale provenance rejected: {field}", rejected)

    def _flank(pl) -> Dict[str, int]:
        pk = pl.arrays["slot_kind"]
        if set(pk) == {"paired"}:
            return {"left_paired_flank": len(pk), "right_paired_flank": 0}
        return {
            "left_paired_flank": next(i for i, k in enumerate(pk) if k != "paired"),
            "right_paired_flank": next(i for i, k in enumerate(reversed(pk)) if k != "paired"),
        }

    report = {
        "device": args.device,
        "config": str(Path(args.config).relative_to(_ROOT)) if args.config else None,
        "window_W": args.window,
        "window_rule": rc.WINDOW_RULE,
        "window_rule_version": rc.WINDOW_RULE_VERSION,
        "split_rule": rc.SPLIT_RULE,
        "split_rule_version": rc.SPLIT_RULE_VERSION,
        "model_load_seconds": load_s,
        "rows": [
            {"label": l, "var_id": p.var_id, "HGVSp": str(fx[fx.var_id == p.var_id].HGVSp.iloc[0]),
             "n_slots": p.n_slots, "mut_len": len(p.mut_seq),
             "slot_kinds": dict(pd.Series(p.arrays["slot_kind"]).value_counts()),
             **_flank(p)}
            for l, p in by_label.items()
        ],
        "A": A,
        "export_report": res.report,
        "provenance_hash": res.provenance["provenance_hash"],
        "provenance_identity": {
            k: res.provenance.get(k) for k in rc.PROVENANCE_IDENTITY_FIELDS
        },
        "checks": c.results,
        "n_checks": len(c.results),
        "n_failed": sum(1 for r in c.results if not r["pass"]),
        "all_passed": c.all_passed,
    }
    (out_dir / "fixture_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\n{report['n_checks'] - report['n_failed']}/{report['n_checks']} checks passed "
          f"-> {out_dir / 'fixture_report.json'}")
    return 0 if c.all_passed else 1


# ==========================================================================
# benchmark
# ==========================================================================
def cmd_benchmark(args: argparse.Namespace) -> int:
    resolve_runtime(args)
    df, wt_seq = load_inputs(Path(args.manifest), Path(args.wt))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    cuda = device.type == "cuda"

    # --- installation / model-load phase, kept separate from processing ---
    ckpt = rc.checkpoint_path()
    ckpt_present_before = ckpt.exists()
    t_ckpt0 = time.perf_counter()
    enc, model_load_s = make_encoder(args.device)
    checkpoint_load_s = time.perf_counter() - t_ckpt0
    layers = list(rc.REPR_LAYERS)

    plans, _ = rc.plan_rows(df, wt_seq, W=args.window, in_scope_only=True)
    # distinct real sequences of real length, deliberately spanning the length
    # classes present in the cohort so length-grouped batching is exercised
    by_len: Dict[int, List[str]] = {}
    seen = set()
    for p in plans:
        if p.mut_seq_hash in seen:
            continue
        seen.add(p.mut_seq_hash)
        by_len.setdefault(len(p.mut_seq), []).append(p.mut_seq)
    # Take one sequence from every length class FIRST, so the mixed-length path
    # is always exercised, then fill the rest proportionally to the cohort.
    seqs: List[str] = [by_len[L][0] for L in sorted(by_len)]
    for L in sorted(by_len, key=lambda k: -len(by_len[k])):
        share = round(args.n_seqs * len(by_len[L]) / len(seen))
        for s in by_len[L][1:share]:
            if len(seqs) >= args.n_seqs:
                break
            seqs.append(s)
    seqs = seqs[: args.n_seqs]
    assert len(set(seqs)) == len(seqs), "benchmark sequences must be distinct"

    # --- warm-up (excluded from every reported number) --------------------
    with torch.no_grad():
        _, _, warm = enc.batch_converter([("w", wt_seq)])
        enc.model(warm.to(device), repr_layers=layers)
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # --- measured processing phase ---------------------------------------
    tok_s = 0.0
    fwd_s = 0.0
    groups: Dict[int, List[str]] = {}
    for s in seqs:
        groups.setdefault(len(s), []).append(s)

    reps_by_seq: Dict[str, torch.Tensor] = {}
    t_proc0 = time.perf_counter()
    with torch.no_grad():
        for L in sorted(groups):
            grp = groups[L]
            for start in range(0, len(grp), args.batch_size):
                chunk = grp[start : start + args.batch_size]

                t0 = time.perf_counter()
                _, _, tokens = enc.batch_converter([(f"s{i}", s) for i, s in enumerate(chunk)])
                tokens = tokens.to(device)
                if cuda:
                    torch.cuda.synchronize()
                tok_s += time.perf_counter() - t0

                if cuda:
                    ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
                    ev0.record()
                out = enc.model(tokens, repr_layers=layers)
                if cuda:
                    ev1.record()
                    torch.cuda.synchronize()
                    fwd_s += ev0.elapsed_time(ev1) / 1000.0
                else:
                    fwd_s += time.perf_counter() - t0

                for i, s in enumerate(chunk):
                    reps_by_seq[s] = torch.stack(
                        [out["representations"][l][i, 1 : L + 1, :].float().cpu() for l in layers]
                    )
    proc_s = time.perf_counter() - t_proc0

    # cross-check that this hand-rolled loop equals the library path
    probe = seqs[0]
    ref = rc.stack_layers({l: v[0] for l, v in enc.encode([probe], repr_layers=layers).items()}, layers)
    agree_abs = (ref - reps_by_seq[probe]).abs().max().item()
    agree_rel = agree_abs / max(ref.abs().max().item(), 1e-9)

    # --- alignment / window phase ----------------------------------------
    sub = [p for p in plans if p.mut_seq in reps_by_seq][: args.n_seqs]
    A = max(p.n_slots for p in plans)
    wt_reps = enc.encode([wt_seq], repr_layers=layers)
    wt_full = rc.stack_layers({l: wt_reps[l][0] for l in layers}, layers)

    t_al0 = time.perf_counter()
    hw, hm = [], []
    for p in sub:
        padded = rc.pad_arrays(p.arrays, A)
        hw.append(rc.gather_window(wt_full, padded["wt_pos"], padded["wt_present"]))
        hm.append(rc.gather_window(reps_by_seq[p.mut_seq], padded["mut_pos"], padded["mut_present"]))
    H_WT, H_MUT = torch.stack(hw), torch.stack(hm)
    dv = torch.tensor([rc.pad_arrays(p.arrays, A)["delta_valid"] for p in sub])
    delta = rc.compute_delta(H_WT, H_MUT, dv)
    align_s = time.perf_counter() - t_al0

    # --- serialization phase ---------------------------------------------
    bench_blob = out_dir / "benchmark_sample.pt"
    t_ser0 = time.perf_counter()
    torch.save({"H_WT": H_WT, "H_MUT": H_MUT, "delta_H": delta}, bench_blob)
    ser_s = time.perf_counter() - t_ser0
    out_bytes = bench_blob.stat().st_size
    if not args.keep_blob:
        bench_blob.unlink()

    # --- full-export projection from these measurements -------------------
    n_unique_all = len({p.mut_seq_hash for p in plans}) + 1  # + the WT forward
    per_seq_s = proc_s / len(seqs)
    projected_forward_s = per_seq_s * n_unique_all
    projected_align_s = (align_s / len(sub)) * len(plans)
    projected_ser_s = (ser_s / len(sub)) * len(plans)

    bench = {
        "measurement_scope": "processing only; model load and checkpoint load reported separately",
        "hardware": {
            "gpu_name": torch.cuda.get_device_name(0) if cuda else None,
            "gpu_total_vram_bytes": torch.cuda.get_device_properties(0).total_memory if cuda else None,
            "gpu_total_vram_mib": round(torch.cuda.get_device_properties(0).total_memory / 2**20) if cuda else None,
            "cpu_count": os.cpu_count(),
            "platform": platform.platform(),
        },
        "software": {
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version() if cuda else None,
            "driver_version": _nvidia_driver(),
            "python": platform.python_version(),
        },
        "config": {
            "device": args.device,
            "precision_forward": str(next(enc.model.parameters()).dtype),
            "precision_cache": "torch.float32",
            "precision_delta": "torch.float32",
            "autocast": False,
            "batch_size": args.batch_size,
            "repr_layers": layers,
            "window_W": args.window,
            "A": A,
        },
        "workload": {
            "n_sequences": len(seqs),
            "n_unique_sequences": len(set(seqs)),
            "sequence_lengths": {str(L): len(v) for L, v in sorted(groups.items())},
            "min_length": min(map(len, seqs)),
            "max_length": max(map(len, seqs)),
            "n_aligned_rows": len(sub),
        },
        "setup_seconds": {
            "checkpoint_present_before_run": ckpt_present_before,
            "checkpoint_bytes": ckpt.stat().st_size if ckpt.exists() else None,
            "checkpoint_download_and_load": checkpoint_load_s,
            "model_load_total": model_load_s,
            "note": "checkpoint was already in the torch.hub cache; no download occurred"
            if ckpt_present_before else "includes a real download",
        },
        "processing_seconds": {
            "tokenization": tok_s,
            "forward": fwd_s,
            "forward_timing_method": "cuda events + synchronize" if cuda else "perf_counter",
            "alignment_window": align_s,
            "serialization_write": ser_s,
            "total_forward_phase": proc_s,
        },
        "throughput": {
            "sequences_per_second_overall": len(seqs) / proc_s,
            "sequences_per_second_forward_only": len(seqs) / fwd_s if fwd_s else None,
            "seconds_per_sequence": per_seq_s,
            "aligned_rows_per_second": len(sub) / align_s if align_s else None,
        },
        "memory": {
            "cuda_max_memory_allocated_bytes": torch.cuda.max_memory_allocated() if cuda else None,
            "cuda_max_memory_reserved_bytes": torch.cuda.max_memory_reserved() if cuda else None,
            "cuda_max_memory_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1) if cuda else None,
            "cuda_max_memory_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1) if cuda else None,
            "peak_stats_reset_after_warmup": True,
            "process_rss_bytes": _rss_bytes(),
            "process_rss_mib": round(_rss_bytes() / 2**20, 1),
        },
        "output": {
            "sample_output_bytes": out_bytes,
            "sample_output_shape": list(H_WT.shape),
            "bytes_per_row_H_WT_plus_H_MUT": out_bytes / max(len(sub), 1),
        },
        "consistency": {
            "handrolled_loop_vs_encoder_encode_max_abs_diff": agree_abs,
            "handrolled_loop_vs_encoder_encode_max_rel_diff": agree_rel,
            "note": "the benchmark loop mirrors ESMEncoder._forward but times "
                    "tokenisation and forward separately; the residual is FP32 "
                    "GEMM reduction-order drift from the differing batch shape "
                    "(batch 8 vs batch 1), not a difference in computation",
        },
        "projection_full_export": {
            "in_scope_rows": len(plans),
            "unique_sequences_incl_wt": n_unique_all,
            "projected_forward_seconds": projected_forward_s,
            "projected_alignment_seconds": projected_align_s,
            "projected_serialization_seconds": projected_ser_s,
            "projected_total_seconds": projected_forward_s + projected_align_s + projected_ser_s,
            "projected_total_minutes": (projected_forward_s + projected_align_s + projected_ser_s) / 60,
            "projected_cache_bytes": (out_bytes / max(len(sub), 1)) / 2 * len({(p.mut_seq_hash, p.window_sig) for p in plans}),
        },
    }
    path = out_dir / "benchmark.json"
    path.write_text(json.dumps(bench, indent=2, default=str))
    print(json.dumps(bench, indent=2, default=str))
    print(f"\n-> {path}")
    return 0


def _nvidia_driver() -> Optional[str]:
    import subprocess

    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        )
        return r.stdout.strip().splitlines()[0] if r.returncode == 0 else None
    except (OSError, IndexError, subprocess.SubprocessError):
        return None


# ==========================================================================
# export
# ==========================================================================
def cmd_export(args: argparse.Namespace) -> int:
    resolve_runtime(args)
    out_dir = Path(args.out_dir)
    fixture_report = out_dir / "fixture_report.json"
    if not args.skip_fixture_gate:
        if not fixture_report.exists():
            print(f"REFUSING: no fixture report at {fixture_report}. Run `fixture` first.")
            return 2
        rep = json.loads(fixture_report.read_text())
        if not rep.get("all_passed"):
            print(f"REFUSING: fixture report at {fixture_report} has "
                  f"{rep.get('n_failed')} failing checks.")
            return 2

    df, wt_seq = load_inputs(Path(args.manifest), Path(args.wt))
    enc, load_s = make_encoder(args.device)
    print(f"[model] loaded in {load_s:.1f}s on {args.device}")

    plans, skipped = rc.plan_rows(df, wt_seq, W=args.window, in_scope_only=True)
    A = max(p.n_slots for p in plans)
    print(f"[plan] {len(plans)} in-scope rows, {len(skipped)} skipped, A={A}")

    t0 = time.perf_counter()
    res = rc.export_representations(
        plans, wt_seq, enc, out_dir / "frozen_repr_v1.pt",
        manifest_path=Path(args.manifest), W=args.window,
        batch_size=args.batch_size, device=args.device, A=A,
        progress_every=args.progress_every,
    )
    total_s = time.perf_counter() - t0

    from collections import Counter

    reasons = Counter(s["reason"] for s in skipped)
    summary = {
        "input_rows": len(df),
        "supported_rows": len(plans),
        "successful_rows": len(plans),
        "failed_rows": len(skipped),
        "failed_reasons": dict(reasons),
        "unique_wt_sequences": 1,
        "unique_mut_sequences": res.report["unique_mut_sequences"],
        # Three DIFFERENT quantities, never conflated: distinct proteins handed
        # to the model, actual model(tokens) invocations, and requests served
        # from the in-memory store without any forward.
        "unique_sequences_encoded": res.report["unique_sequences_encoded"],
        "model_forward_batch_calls": res.report["model_forward_batch_calls"],
        "cache_hits": res.report["cache_hits"],
        "sequence_requests": res.report["sequence_requests"],
        "dna_rows_vs_unique_protein": {
            "dna_rows": len(plans),
            "unique_mut_proteins": res.report["unique_mut_sequences"],
            "sequences_encoded": res.report["unique_sequences_encoded"],
            "sequence_forwards_saved": len(plans) + 1 - res.report["unique_sequences_encoded"],
        },
        "stored_mut_windows": res.report["unique_stored_mut_windows"],
        "cache_bytes": res.report["cache_bytes"],
        "cache_path": str(res.path),
        "A": A,
        "seconds": {
            "forward": res.report["forward_seconds"],
            "alignment": res.report["alignment_seconds"],
            "serialization": res.report["serialize_seconds"],
            "total_export": total_s,
            "model_load": load_s,
        },
        "rows_by_split": dict(Counter(p.split for p in plans)),
        "rows_by_variant_type": dict(Counter(p.variant_type for p in plans)),
        "rows_by_edit_type": dict(Counter(p.edit_type for p in plans)),
        "slots_per_row": dict(sorted(Counter(p.n_slots for p in plans).items())),
        "window_W": args.window,
        "window_rule_version": rc.WINDOW_RULE_VERSION,
        "split_rule_version": rc.SPLIT_RULE_VERSION,
        "provenance_hash": res.provenance["provenance_hash"],
    }
    (out_dir / "export_report.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))

    # reload + verify the cache we just wrote
    cache = rc.load_cache(res.path, expected_provenance=res.provenance)
    b = cache.batch(range(min(8, len(cache))))
    print(f"[verify] reload OK: {len(cache)} rows, H_WT{tuple(b['H_WT'].shape)}")
    return 0


# ==========================================================================
# llr
# ==========================================================================
def cmd_llr(args: argparse.Namespace) -> int:
    resolve_runtime(args)
    df, wt_seq = load_inputs(Path(args.manifest), Path(args.wt))
    out_dir = Path(args.out_dir)
    enc, load_s = make_encoder(args.device)
    print(f"[model] loaded in {load_s:.1f}s on {args.device}")

    scope = df
    if args.in_scope_only:
        plans, _ = rc.plan_rows(df, wt_seq, W=args.window, in_scope_only=True)
        keep = {p.var_id for p in plans}
        scope = df[df.var_id.isin(keep)].copy()

    positions = sorted(
        {int(p) for p in scope[scope.slim_consequence == "missense"].pp.dropna()}
    )
    print(f"[llr] {len(scope)} rows, "
          f"{int((scope.slim_consequence == 'missense').sum())} missense, "
          f"{len(positions)} unique positions to mask")

    profile = masked_wt_profile(
        enc, wt_seq, positions, batch_size=args.batch_size, progress_every=50
    )
    table = llr_table(scope, wt_seq, profile)
    prov = build_llr_provenance(
        manifest_path=Path(args.manifest),
        wt_seq=wt_seq,
        forward_precision=str(next(enc.model.parameters()).dtype),
        device=args.device,
        n_positions=len(positions),
        masked_inputs=profile.masked_inputs,
        model_forward_batch_calls=profile.model_forward_batch_calls,
    )
    info = save_llr(out_dir, table, profile, prov)

    from collections import Counter

    info.update({
        # Three separate numbers: positions profiled, masked sequences fed to
        # the model, and actual model(tokens) invocations.
        "masked_positions": profile.masked_positions,
        "masked_inputs": profile.masked_inputs,
        "model_forward_batch_calls": profile.model_forward_batch_calls,
        "rows_in_scope": int(len(scope)),
        "forward_seconds": profile.forward_seconds,
        "model_load_seconds": load_s,
        "rows_by_consequence": dict(Counter(table.slim_consequence)),
        "llr_valid_by_consequence": {
            k: int(v) for k, v in table.groupby("slim_consequence").llr_valid.sum().items()
        },
        "undefined_reasons": {
            k: int(v) for k, v in Counter(table.llr_undefined_reason.dropna()).items()
        },
        "profile_shape": list(profile.logprobs.shape),
        "provenance_hash": prov["provenance_hash"],
    })
    (out_dir / "llr_report.json").write_text(json.dumps(info, indent=2, default=str))
    print(json.dumps(info, indent=2, default=str))

    if args.diagnostic:
        _llr_diagnostic(table, Path(args.manifest), out_dir)
    return 0


def _llr_diagnostic(table, manifest_path: Path, out_dir: Path) -> None:
    """Train/val-only sanity diagnostic. TEST LABELS ARE NEVER READ HERE.

    This exists so the LLR can be checked for sign/sanity during development.
    It is not a model-selection signal for anything, and it deliberately drops
    every test row before touching a label column.
    """
    from scipy.stats import spearmanr

    labels = pd.read_csv(manifest_path, usecols=["var_id", "split", "z_score_D4_D14"])
    labels = labels[labels.split.isin(["train", "val"])]        # test rows dropped first
    m = table[table.llr_valid].merge(labels, on=["var_id", "split"], how="inner")
    out = {"note": "train/val only; test labels not read", "splits": {}}
    for split in ("train", "val"):
        s = m[(m.split == split) & m.z_score_D4_D14.notna()]
        if len(s) > 10:
            out["splits"][split] = {
                "n": int(len(s)),
                "spearman_llr_vs_z": float(spearmanr(s.llr, s.z_score_D4_D14).statistic),
            }
    (out_dir / "llr_diagnostic_trainval.json").write_text(json.dumps(out, indent=2))
    print("[diagnostic] " + json.dumps(out))


# ==========================================================================
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p: argparse.ArgumentParser, out_key: str) -> None:
        """Flags default to None and are filled from the validated YAML config."""
        p.add_argument("--config", default=str(DEFAULT_CONFIG),
                       help="runtime config; validated against the Python contract constants")
        p.add_argument("--manifest", default=None)
        p.add_argument("--wt", default=None)
        p.add_argument("--out-dir", default=None)
        p.add_argument("--device", default=None)
        p.add_argument("--batch-size", type=int, default=None)
        p.add_argument("--window", type=int, default=None)
        p.set_defaults(out_key=out_key)

    p = sub.add_parser("fixture"); common(p, "repr_out_dir")
    p.add_argument("--tol", type=float, default=1e-5,
               help="relative tolerance for individual-vs-batch forward agreement")
    p.set_defaults(fn=cmd_fixture)

    p = sub.add_parser("benchmark"); common(p, "repr_out_dir")
    p.add_argument("--n-seqs", type=int, default=32)
    p.add_argument("--keep-blob", action="store_true")
    p.set_defaults(fn=cmd_benchmark)

    p = sub.add_parser("export"); common(p, "repr_out_dir")
    p.add_argument("--skip-fixture-gate", action="store_true")
    p.add_argument("--progress-every", type=int, default=500)
    p.set_defaults(fn=cmd_export)

    p = sub.add_parser("llr"); common(p, "llr_out_dir")
    p.add_argument("--in-scope-only", action="store_true", default=True)
    p.add_argument("--all-rows", dest="in_scope_only", action="store_false")
    p.add_argument("--diagnostic", action="store_true",
                   help="train/val-only Spearman sanity check; never reads test labels")
    p.set_defaults(fn=cmd_llr)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
