# DeepRAD51C — P1 (Data) + P4 (Baseline)

> ⚠️ **PRIVATE / EMBARGOED.** The RAD51C SGE data is unpublished and embargoed
> until the Sanger paper is out. The files in `data/` derive from it. **Keep this
> repo private (team only). Do not make it public or redistribute the data.**

Owner: 구민선 (P1 Data Engineering + P4 Baseline).

## What's here

| File | What it is |
|---|---|
| `build_annotation.py` | builds the per-residue functional annotation table → `data/rad51c_residue_annotation.csv` [376×10] |
| `dataset.py` | loads SGE, builds mutant sequences, **position-based train/val/test split**, → `data/split_manifest.csv` |
| `baseline_llr.py` | P4 ESM-2 zero-shot LLR baseline → `data/baseline_llr.csv` |
| `data/split_manifest.csv` | **the shared deliverable** (see below) |
| `data/wt_sequence.txt` | RAD51C WT protein sequence (376 aa) |
| `data/rad51c_residue_annotation.csv` | the [376×10] annotation table |

## Setup (each teammate, once)

```bash
git clone <this-private-repo-url>
cd rad51c_project
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

To regenerate anything yourself, also clone the raw data next to the scripts:
```bash
git clone https://github.com/team113sanger/RAD51C_SGE.git
```
(You only need this to re-run `dataset.py`/`baseline_llr.py`. To just *use* the
split, `data/split_manifest.csv` is enough.)

## How to use `data/split_manifest.csv` (P2 / P3)

One row per variant (5,887). Columns:

| column | meaning |
|---|---|
| `var_id` | **join key** (e.g. `chr17_58692647_C_G`) — attach your embeddings on this |
| `split` | `train` / `val` / `test` — **use this, do not re-split** |
| `slim_consequence` | missense / synonymous / codon_deletion / inframe del-ins |
| `pp` | protein position (1-based start residue) |
| `ref_aa`, `alt_aa` | WT → mutant amino acid (blank for indels) |
| `HGVSp` | protein change, e.g. `p.Arg2Gly`, `p.Ile64del` |
| `z_score_D4_D14` | **regression label** (SGE functional score) |
| `functional_classification` | **classification label** |
| `mut_seq` | **mutant protein sequence** — feed this to ESM-2 / ESMFold |

**Two rules:**
1. Join your features to variants on **`var_id`**.
2. Train/evaluate using the **`split`** column. Never make your own split — it's
   position-based (no codon leakage) and shared so everyone's numbers compare.

## Split & scope

- Position-based 70/15/15 (seed=42): **train 4124 / val 881 / test 882**, no
  protein position shared across splits (asserted in `dataset.py`).
- Scope = exclude `intron`, `frameshift`, `stop_gained`, `UTR`. Splice / start_lost
  / stop_lost are dropped (no protein-sequence change). Buildable = missense +
  synonymous + codon_deletion + inframe del/ins.

## P4 baseline to beat (ESM-2 150M, test split)

Spearman ρ = **0.59**, Pearson r = 0.67, ROC-AUC (depleted vs unchanged) = **0.92**.
The trained model must beat these.
