"""Synthetic CPU fixtures shared by the test suite and train_stage1.py --dry-run.

No real ESM-2 weights and no network access are used anywhere in this
module -- FakeFrozenEncoder (cache.py) stands in for the frozen backbone.
"""

from __future__ import annotations

from dataclasses import dataclass

from .cache import FakeFrozenEncoder, build_cache_from_manifest
from .dataset import build_cohort, join_cohort_with_cache

SYNTHETIC_WT = "MAKGTFRLEQVDLNSFPLSPRVKLVSAGFTQAELLEHKPSDVGISKAEALQTLR"  # 55 aa, no real biology


@dataclass
class SyntheticFixture:
    wt_seq: str
    manifest_rows: list[dict]
    cache: "object"          # RawStage1Cache
    cohort_entries: list[dict]
    skipped: list[dict]


def _row(var_id, cons, pp, ref_aa=None, alt_aa=None, hgvsp=None, mut_seq=None, split="train", label=0.0):
    return {
        "var_id": var_id, "split": split, "slim_consequence": cons, "pp": pp,
        "ref_aa": ref_aa, "alt_aa": alt_aa, "HGVSp": hgvsp, "mut_seq": mut_seq,
        "z_score_D4_D14": label,
    }


def make_synthetic_manifest(wt_seq: str = SYNTHETIC_WT) -> list[dict]:
    rows = []

    # missense at position 10 (1-based)
    p = 10
    ref, alt = wt_seq[p - 1], "W" if wt_seq[p - 1] != "W" else "Y"
    mut = wt_seq[:p - 1] + alt + wt_seq[p:]
    rows.append(_row("syn_missense_1", "missense", p, ref_aa=ref, alt_aa=alt, mut_seq=mut, label=1.5))

    # synonymous at position 20 -- mutant protein == WT
    p = 20
    ref = wt_seq[p - 1]
    rows.append(_row("syn_synonymous_1", "synonymous", p, ref_aa=ref, mut_seq=wt_seq, label=-0.2))

    # single-residue deletion (codon_deletion) at position 15
    p = 15
    mut = wt_seq[:p - 1] + wt_seq[p:]
    rows.append(_row("syn_del1_1", "codon_deletion", p, hgvsp=f"NP_TEST:p.Xaa{p}del", mut_seq=mut, label=-3.0))

    # multi-residue deletion, positions 30-33 (4 residues)
    s, e = 30, 33
    mut = wt_seq[:s - 1] + wt_seq[e:]
    rows.append(_row("syn_del4_1", "clinical_inframe_deletion", s,
                      hgvsp=f"NP_TEST:p.Xaa{s}_Xaa{e}del", mut_seq=mut, label=-5.0))

    # single-residue duplication (insertion) at position 25
    p = 25
    mut = wt_seq[:p] + wt_seq[p - 1:p] + wt_seq[p:]
    rows.append(_row("syn_dup_1", "clinical_inframe_insertion", p,
                      hgvsp=f"NP_TEST:p.Xaa{p}dup", mut_seq=mut, label=0.8))

    # delins: delete 5, insert 1 (mirrors the required p.Leu338_Lys342delinsGln shape)
    s, e, ins = 35, 39, "Q"
    mut = wt_seq[:s - 1] + ins + wt_seq[e:]
    rows.append(_row("syn_delins_1", "clinical_inframe_deletion", s,
                      hgvsp=f"NP_TEST:p.Xaa{s}_Xaa{e}delins{('Gln' if ins == 'Q' else ins)}",
                      mut_seq=mut, label=-8.0))

    # unsupported: manifest row simulating a consequence outside scope
    rows.append(_row("syn_unsupported_1", "frameshift", 40, mut_seq=None, label=float("nan")))

    return rows


def make_synthetic_fixture(hidden_dim: int = 16, layers: list[int] = (33,)) -> SyntheticFixture:
    wt_seq = SYNTHETIC_WT
    rows = make_synthetic_manifest(wt_seq)
    cohort = build_cohort(rows, wt_seq)

    encoder = FakeFrozenEncoder(hidden_dim=hidden_dim, seed=0)
    cache_rows = [{"var_id": e["var_id"], "mut_seq": e["row"]["mut_seq"]} for e in cohort.supported]
    cache = build_cache_from_manifest(cache_rows, wt_seq, list(layers), encoder)

    cohort2 = join_cohort_with_cache(cohort, cache)
    return SyntheticFixture(
        wt_seq=wt_seq, manifest_rows=rows, cache=cache,
        cohort_entries=cohort2.supported, skipped=cohort2.skipped,
    )
