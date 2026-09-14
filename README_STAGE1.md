# Stage 1 — Frozen-ESM Representation Comparison (`paired_delta` / `branched_projection` / `unified_reference_delta`)

This is a separate path from the legacy M1–M4 concat models (`model.py`,
`train.py`, `dataset.py`, `src/embeddings/*`, `configs/{base,m1,m2,m3,m4}.yaml`).
Nothing under M1–M4 was touched; Stage 1 lives entirely under `src/stage1/`,
`configs/stage1/`, `train_stage1.py`, `dump_stage1_cache.py`, and
`tests/test_stage1_*.py`.

**Status of this build: code + CPU-synthetic verification only.** No real
ESM-2 extraction, no real training, no `optimizer.step()`, no checkpoint
download happened in producing this. See "What has and hasn't run" at the
bottom.

## 1. Why a separate path

M1–M4 compare a *pooled, cached, single-vector* WT-difference embedding
(`diff_emb_raw.pt`, shape `(1280,)` per variant) feeding a concat+residual
backbone. Stage 1 compares three ways of turning **unpooled, per-residue**
WT/MUT hidden states into a sequence of content tokens that a small
attention-pooling head consumes — a different representation question, with
its own cache format (unpooled, so the legacy `diff_emb_raw.pt` / sweep
caches are **not** reusable here; see §5).

## 2. Code layout

```
src/stage1/
  schema.py         constants shared by every other module (slot kinds, dims, versions)
  alignment.py       HGVS -> (u,d,inserted,m) normalization, WT<->MUT slot map, window selection
  positional.py       fixed sinusoidal PE (anchor-relative coord + insertion rank)
  metadata.py         common edit-metadata featurizer + train-only MetaScaler
  cache.py             raw per-residue ESM hidden-state cache (schema, build, FakeFrozenEncoder)
  window.py            alignment + cache -> per-sample raw tensors (SampleTokens)
  modules.py            ProjectionMLP, the 3 ContentBuilders, layer embedding wiring,
                        ConstantQueryPooling, SequenceHead
  model.py               Stage1Model (assembles everything), freeze_for_stage2()
  dataset.py              Stage1Dataset, cohort building/validation, collate
  checkpoint.py            save/load, Stage 2 hand-off contract
  split_audit.py           edited-span vs. shipped-split overlap audit
  config.py                yaml config loading (reuses src/config_loader.py's `extends:`)
  engine.py                 train/eval loop, optimizer, target/meta scaling (NOT executed)
  experiment_plan.py        3-model x window x fold x seed plan generator
  synthetic.py               CPU synthetic fixture shared by tests and --dry-run

configs/stage1/{base,paired_delta,branched_projection,unified_reference_delta}.yaml
train_stage1.py        entry point: --dry-run / --audit-split / --plan / --train (guarded)
dump_stage1_cache.py    raw-cache builder entry point (not run — needs the real ESM-2 checkpoint)
tests/test_stage1_alignment.py       alignment + the two required real-data fixtures
tests/test_stage1_window_pe.py       window/PE/attention-mask CPU checks
tests/test_stage1_models.py           content-builder routing, param counts, seeded init, freeze
tests/test_stage1_dataset_cache.py     cache schema, var_id join validation
tests/test_stage1_checkpoint.py         Stage 1 -> Stage 2 checkpoint contract
tests/test_stage1_engine_plan.py         train-only preprocessing, plan generation, split audit,
                                          "import/--help never trains" guarantee
```

If there was already a suitable implementation of any of the above, it would
have been extended rather than duplicated — there wasn't (M1–M4's modules
are pooled-vector-shaped and don't fit this schema), so this is new code.

## 3. The three models

All three read the SAME window of aligned WT/MUT slots and the SAME edit
metadata token; they differ only in how a slot's per-layer content vector
`c_{l,a}` (dim 128) is computed from the frozen ESM hidden states.

| | Paired content | One-sided content | Projections |
|---|---|---|---|
| **A `paired_delta`** | `P_Δ(ΔH)` | not used (masked out of attention) | 1 |
| **B `branched_projection`** | `P_Δ(ΔH)` | `P_WT(H_WT)` / `P_MUT(H_MUT)` per slot kind | 3 |
| **C `unified_reference_delta`** | `P_shared([R; ΔH; flags])`, `R=H_WT` | `P_shared([R; ΔH; flags])`, `R=H_WT` or `H_MUT` | 1 |

Every projection is `Linear(in, 32) -> GELU -> Linear(32, 128)` (bias=True on
both layers, bottleneck_dim=32 by default, configurable). At `D_esm=1280`:

| model | content-projection params | formula |
|---|---:|---|
| paired_delta | **45,216** | `32*1280+32 + 32*128+128` |
| branched_projection | **135,648** | `3 x` the row above |
| unified_reference_delta | **86,368** | `in_dim = 2*1280+6 = 2566` |

`python train_stage1.py --dry-run` prints these numbers plus the full-model
total (content projection + layer embedding + metadata encoder + query/
temperature + sequence head) for the model(s) requested — verified to match
exactly in `tests/test_stage1_models.py::test_content_projection_param_counts_match_task_example`.

**What is genuinely controlled across A/B/C:** ESM checkpoint+layers, window
rule, alignment/edit-map, PE, layer embedding, metadata token+encoder,
pooling, sequence head, loss/optimizer/model-selection rule, seeds, split.
**What is NOT controlled, and the results must be read accordingly:** B has
3x A's content-projection capacity and sees one-sided residue content that A
never does; C shares one projection like A but sees WT context + one-sided R
+ flags that A never does. **A -> B is confounded by capacity AND
information. B <-> C differ in projection sharing AND how paired context is
presented (Δ-only vs. WT+Δ) AND how flags reach the model.** This is reported
as a three-way representation-design comparison, not a clean single-factor
ablation — the task explicitly asked that this not be oversold as one.

## 4. Common edit metadata token

One extra token per sample (not per layer), built from sequence/edit-derived
features only — never z-score, functional class, split id, or var_id:

`raw_meta_features` (10-dim): `[one-hot(missense,synonymous,deletion,insertion,delins)]
+ [u, d, m, wt_len, mut_len]`. The one-hot half is left alone; the 5 numeric
values are standardized with a `MetaScaler` **fit on the training split
only** (mirrors `train.py`'s `standardize_struct`). `translation_status` is
recorded on `NormalizedEdit` for bookkeeping but deliberately excluded from
the trainable feature vector — it is constant across the currently supported
cohort, so it would be a dead input dimension.

The metadata token gets its own `ProjectionMLP(10 -> 32 -> 128)` (same family
as the content projections, own weights) and is appended to the K/V sequence
with **no** positional encoding and **no** layer embedding added.

## 5. Alignment / coordinate / mask schema

Every edit is normalized to `MUT = WT[:u] + inserted_seq + WT[u+d:]` (`u` =
preserved WT prefix length, `d` = WT residues removed, `inserted_seq`/`m` =
new residues). Missense and synonymous are the special case `d=1,m=1`
(synonymous: `inserted_seq == WT[u]`, giving an exact `Δ=0` once both
sequences hit the same frozen ESM — no epsilon tolerance needed).

Slot mapping is **index-based only**, never amino-acid-identity-based:

* WT `[0,u)` <-> MUT `[0,u)`: paired
* `d==1 and m==1`: the one edited residue is ALSO paired (this is what makes
  a validated substitution paired instead of wt_only+mut_only)
* otherwise: WT `[u,u+d)` -> `wt_only`, MUT `[u,u+m)` -> `mut_only`
* WT suffix `[u+d, wt_len)` <-> MUT `[u+m, mut_len)` (MUT idx = WT idx + m-d), paired

Verified against the two real fixtures from the task spec, using the actual
`data/wt_sequence.txt` and `data/split_manifest.csv` rows already in this
repo (`tests/test_stage1_alignment.py`):
`p.Leu338_Lys342delinsGln` (`chr17_58732531_TGTTTCAAATCA_`) and
`p.Lys186dup` (`chr17_58696842__AAA`). A third test in the same file
(`test_real_manifest_full_cohort_supported_or_reported`) runs
`normalize_edit` + `validate_reconstruction` against **every** one of the
5,887 shipped manifest rows: 5,887 supported, 0 skipped.

Mask table:

| slot | wt_present | mut_present | delta_valid | token_valid |
|---|---:|---:|---:|---:|
| paired | 1 | 1 | 1 | 1 |
| wt_only | 1 | 0 | 0 | 1 |
| mut_only | 0 | 1 | 0 | 1 |
| pad (batch only) | 0 | 0 | 0 | 0 |

`attention_valid` is separate from `token_valid`: model A uses
`token_valid & delta_valid` (paired-only attention); B/C use `token_valid`
directly. The metadata token's validity is always 1. Coordinates
(`wt_pos`/`mut_pos`) are stored 1-based with sentinel `0` for "absent" and
are only converted to a 0-based index after checking presence — never by
reinterpreting a `-1` sentinel.

**Positional encoding convention chosen for THIS build** (documented as our
choice, not claimed to be an already-fixed formula from prior docs): 128 =
64 dims signed-sinusoidal encoding of `(WT position - u)` (0 for mut_only
slots, by definition of the boundary anchor) + 64 dims signed-sinusoidal
encoding of 1-based insertion rank (zeroed for every non-mut_only slot). Same
coordinate always encodes identically regardless of window radius `W`
(`tests/test_stage1_alignment.py::test_anchor_relative_coordinates_independent_of_window_radius`,
`tests/test_stage1_window_pe.py::test_pe_same_wt_position_identical_encoding_across_window_radii`).

## 6. Window rule

`window_radius=W` (config/CLI, unrestricted): missense/synonymous take the
single paired event slot + up to `W` paired flanks each side; indels take
every event slot (`wt_only`+`mut_only`, however many the edit has) + up to
`W` paired flanks each side, clipped at protein termini. `W=0` on a pure
indel yields **zero valid attention tokens for model A** except the metadata
token — this is the documented "metadata-only edge case"
(`train_stage1.py --dry-run` and `tests/test_stage1_window_pe.py::test_metadata_only_edge_case_window_radius_zero`
both exercise it and confirm no NaN / a well-defined softmax over the single
metadata token).

## 7. Cache

`RawStage1Cache` stores **unpooled, per-residue** frozen-ESM hidden states:
one shared `H_wt` (forwarded once) + one `H_mut` per variant (grouped by
length, mirroring `src/embeddings/diff_embedder.py`'s WT-broadcast trick),
for whichever `layers` were requested. All three model modes slice their own
window/content out of this **same** cache — no separate extraction per
model. Schema-versioned (`stage1-raw-v1`); loading a cache with a different
version raises immediately rather than silently reinterpreting it.

**The legacy `data/diff_emb_raw.pt` and `data/sweep/*.pt` caches are pooled,
single-vector-per-variant caches and are not compatible with this schema —
do not point Stage 1 at them.**

Building the real cache requires downloading `esm2_t33_650M_UR50D` and is
**not done by this task**:

```bash
python dump_stage1_cache.py --manifest data/split_manifest.csv \
    --wt-seq data/wt_sequence.txt --out data/stage1/raw_cache.pt --layers 33
```

For CPU tests/dry-run, `src/stage1/cache.py::FakeFrozenEncoder` stands in for
`ESMEncoder` with the exact same `.encode()` contract — deterministic,
download-free, and it produces bit-identical WT/MUT vectors for identical
sequences (so synonymous variants get an exact `Δ=0`, same as the real ESM
would give for two identical forward passes in eval mode).

## 8. Split policy and the edited-span audit

The shipped `data/split_manifest.csv` split (position-based, grouped by each
variant's own anchor `pp`) is reused as-is — Stage 1 does not re-split.
`train_stage1.py --audit-split` checks something the shipped split does
**not** guarantee: that an indel's full **edited span** (not just its
anchor) stays inside one split. Running it against the real data in this
repo right now:

```
[audit-split] cohort: 5887 supported, 0 skipped
[audit-split] indels audited: 359
[audit-split] start-position overlaps (should be 0): 0
[audit-split] edited-span overlaps: 2
    OVERLAP {'var_id': 'chr17_58732531_TGTTTCAAATCA_', 'position': 342, 'own_split': 'train', 'position_split': 'val'}
    OVERLAP {'var_id': 'chr17_58734138__TGT', 'position': 352, 'own_split': 'test', 'position_split': 'val'}
```

This is reported, not auto-fixed: `chr17_58732531_TGTTTCAAATCA_` (the
delinsGln variant, in `train`) has WT position 342 inside its deleted span,
and some other variant anchored at 342 landed in `val`; similarly for the
`Val351dup` variant (`test`) and WT position 352 (`val`). Two overlaps out of
359 indels audited. `--strict-edited-span` refuses to run further without an
explicit `--edited-span-policy {exclude,group}`, since there is currently no
common policy decided for these two rows.

## 9. Stage 2 hand-off

`Stage1Model.freeze_for_stage2()` sets `requires_grad=False` on every
parameter, calls `.eval()`, and makes the model's own `.train()` a
permanent no-op (a Stage 2 model's `train()` sweep cannot silently
re-activate Stage 1's dropout). `src/stage1/checkpoint.py` saves/loads:
`model_mode`, resolved `cfg`, `state_dict`, `MetaScaler` state, `y_mean`/
`y_std`, and a `reference` dict (`esm_checkpoint`, cache schema version, WT
hash, alignment version) — **never** the frozen ESM backbone weights
themselves. Loading with the wrong `model_mode`, or against an incompatible
`reference`, raises immediately (`CheckpointModeMismatchError` /
`CheckpointReferenceMismatchError`) rather than partially loading.

The metadata token is always present at K/V index `Lyr*A` (`out["meta_token_index"]`
in `Stage1Model.forward(..., return_extras=True)`), so a Stage 2 model
consuming `K`/`V`/`attention_valid` knows exactly which position it is.

## 10. Commands available now (CPU, no ESM, no network)

```bash
python train_stage1.py --dry-run                         # CPU synthetic forward + param counts, all 3 modes
python train_stage1.py --dry-run --model paired_delta      # just one mode
python train_stage1.py --audit-split                       # read-only split/edited-span audit (real data)
python train_stage1.py --plan --windows 5 10 20 --seeds 42 43 44   # writes runs/stage1/experiment_plan.json, trains nothing
python -m pytest tests/test_stage1_*.py -q                  # 63 tests, all CPU/synthetic
```

`--plan` never overwrites an existing run: any `out_dir` that already has a
`metrics.json` is marked `"skip_existing": true` in the plan instead.

## 11. Commands that require real data prep first (NOT run by this task)

```bash
# 1) build the raw cache (downloads ESM-2 650M -- not done here)
python dump_stage1_cache.py --manifest data/split_manifest.csv \
    --wt-seq data/wt_sequence.txt --out data/stage1/raw_cache.pt --layers 33

# 2) a single explicit training run
python train_stage1.py --model paired_delta --window 10 --seed 42 \
    --cache data/stage1/raw_cache.pt --out runs/stage1/paired_delta/W10/shipped_split/seed42 --train

# 3) executing the full generated plan (explicit, separate from --plan itself)
python train_stage1.py --plan --windows 5 10 20 --execute-plan   # currently refuses with a clear message
```

## 12. What has and hasn't run in producing this

**Done:** all code above; `python -m pytest tests/ -q` (114 passed: 63 new
Stage 1 tests + 51 pre-existing tests, run in a fresh project `.venv` with
CPU-only torch); `train_stage1.py --dry-run` (all 3 modes, forward+backward
connectivity, no `optimizer.step()`); `train_stage1.py --audit-split`
against the real shipped manifest; `train_stage1.py --plan`.

**Not done (explicitly out of scope this session):** any real ESM-2 download
or extraction, `dump_stage1_cache.py` for real, any `--train` or
`--execute-plan` run, any test-split evaluation, any git commit/push.

**Before real training can start:** (1) build the raw cache with
`dump_stage1_cache.py` (needs the ESM-2 650M checkpoint), (2) decide the
`--edited-span-policy` for the 2 flagged overlaps if strict edited-span
evaluation is wanted, (3) pick window radii to sweep via `--plan --windows`,
(4) run `--train` per `(model, window, fold, seed)` from the generated plan.
