# ESM progress log

## Status summary

- Completed: environment verification for the active repo and venv
- Completed: repo state and instruction file review
- Completed: creation of the reproducible context notes and the environment probe script
- Completed: Task A gate; no package reinstall, no Python/PyTorch change
- Completed: Task B canonical row-level + alignment-slot schema
- Completed: **Task C — frozen ESM-2 650M representation cache + masked-WT 650M LLR**
- Completed: **Task C correction round** — the split rule and the indel window
  definition were both wrong; see *Task C correction round* below. All Task B/C
  artifacts were regenerated from the corrected code.
- Deferred: any adaptation (LoRA/Q), Bradley-Terry, Mixout, KL, stage 1/2 training

Storage gate: the home volume was expanded from 4.94 GB total / 0.59 GB free to
**100 GB total / 96 GB available**, so the Task C checkpoint gate is lifted. The
`esm2_t33_650M_UR50D` checkpoint was already present in the torch.hub cache
(`~/.cache/torch/hub/checkpoints/esm2_t33_650M_UR50D.pt`, 2,604,537,549 bytes);
**no download was performed in this session**.

---

## Task C correction round

Two scientific-contract defects were found by review and corrected. Nothing else
about Task C was redesigned: repo-native `esm2_t33_650M_UR50D`, layers
`[31,32,33]`, equal-length batching / `encode_by_length`, separate
`wt_pos`/`mut_pos` gathering, synonymous exact `delta=0`, FP32 delta, masked-WT
650M LLR/profile, full-vocabulary `log_softmax` then AA20 selection, the
feature/target separation and the provenance/stale-cache checks are all
unchanged.

### 1. Cross-span exclusion was reading the attention context, not the edit

**Cause.** `build_variant_record()` derived `cross_span` from
`rec["wt_pos"] + rec["mut_pos"]`. Those arrays are the *alignment* — they carry
the paired boundary/context residues on both sides of the edit, and MUT
coordinates. The split rule was therefore asking "does this variant's alignment
window touch more than one split?", not "does the edit itself cross a split?".

Because the split is strictly per-position (every one of the 375 populated WT
positions belongs to exactly one split, verified), a one-residue deletion at
position *p* was tested on `{p-1, p, p+1}` — so a deletion wholly inside train
was excluded whenever either paired flank happened to be val or test.

Of the old 229 exclusions:

| | count | what actually triggered it |
|---|---|---|
| one-residue deletions | **227** | a paired flank at `p-1`/`p+1` in another split; the deleted residue itself was always in exactly one split |
| `p.Leu338_Lys342delinsGln` | 1 | genuine — the edited WT span 338–342 does cross train/val |
| `p.Val351dup` | 1 | genuine — the insertion boundary 351/352 does cross test/val |

Example: `chr17_58692647_CGC_` (`p.Arg2del`, train) was excluded because its
alignment covers WT 1, 2, 3 and WT 3 is val. The deleted residue is WT 2, train.

**Fix.** `_edit_geometry()` now derives `edited_wt_positions`, and that list is
the only input to the rule (`SPLIT_RULE` / `SPLIT_RULE_VERSION =
split_rule_v2_edited_wt_span_only`, recorded in every provenance record):

| edit | positions the split rule reads |
|---|---|
| deletion | the directly deleted WT span `start..end` |
| delins | the directly edited WT span `start..end` |
| insertion | the WT insertion boundary `(start, start+1)` |
| duplication | the WT boundary the copy is inserted at, `(end, end+1)` |

Paired context/window residues are excluded, and MUT coordinates are never
interpreted as WT split positions. Each record now also carries
`split_positions_considered` and `split_position_splits` so the decision is
auditable per row.

### 2. The indel window double-counted the boundary pairs

**Cause.** `build_window_alignment()` took Task B's edit core — which already
contains the left and right *paired boundary* residues — and then appended a
further `W=10` paired slots outside it. Every indel window was therefore two
slots too wide, and the "10 flank residues" the contract promised were really 11
on each side.

**Fix.** The window is now built from the edit geometry
(`left_boundary_wt` / `right_boundary_wt` / `right_offset`) rather than by
padding a core that already contains boundaries. Exactly `W` paired residues are
emitted on each side, counted *inward* from the edit's own boundaries, with the
boundary residues inside those `W`
(`WINDOW_RULE_VERSION = window_v2_flank_from_edit_boundaries`):

| case | window | before |
|---|---|---|
| missense / synonymous | 10 + 1 + 10 = **21** | 21 (unchanged) |
| one-residue deletion | 10 + 1 `wt_only` + 10 = **21** | 23 |
| one-residue duplication | 10 + 1 `mut_only` + 10 = **21** | 24 |
| `p.Leu338_Lys342delinsGln` | 10 + 5 `wt_only` + 1 `mut_only` + 10 = **26** | 28 |

Fewer slots are emitted only where a terminus truncates the flank — e.g.
`p.Arg2Gly` gives WT 1..12 (12 slots) and `p.Leu376Ile` gives WT 366..376 (11).
Tests assert the exact flank/event budget, not `>= 10`.

Task B's core is *not* re-derived by the window builder, so the two derivations
are now an independent cross-check of each other: `core_offset_in_window()`
requires the Task B core to reappear verbatim and contiguously inside the
geometry-built window, and `plan_rows()` refuses any row where it does not.

### 3. Corrected scope

`python -m src.embeddings.variant_map --out-dir data/variant_audit`:

| | old (wrong rule) | corrected |
|---|---|---|
| total rows | 5,887 | 5,887 |
| supported (`in_eval_scope`) | 5,658 | **5,885** |
| excluded | 229 | **2** |
| excluded by edit type | deletion 227, delins 1, dup 1 | delins 1, duplication 1 |
| supported by split | train 3,997 / val 829 / test 832 | **train 4,123 / val 881 / test 881** |
| supported in-frame indels | train 127 / val 0 / test 3 | **train 253 / val 52 / test 52** |

The two remaining exclusions, both genuine and both `cross_span_edit`:

| var_id | HGVSp | edited WT positions | splits |
|---|---|---|---|
| `chr17_58732531_TGTTTCAAATCA_` | `p.Leu338_Lys342delinsGln` | 338, 339, 340, 341, 342 | train {338–341}, val {342} |
| `chr17_58734138__TGT` | `p.Val351dup` | 351, 352 | test {351}, val {352} |

Supported examples: `p.Arg2del` edits WT {2} = train → kept; `p.Gly3del` edits
WT {3} = val → kept; `p.Lys186dup` inserts at boundary {186, 187}, both train →
kept. All 356 real pure-deletion rows delete a single residue, so under the corrected
rule none of them can be cross-span — asserted row by row in the tests.
`audit.json` now carries `split_rule`, `split_rule_version` and a per-row
`indel_split_decisions` block for all 359 in-frame indels.

### 4. Reporting: sequence counts are not forward counts

The old `actual_esm_forwards=2500` was a *sequence* count. `ESMEncoder` now
counts real `model(tokens)` invocations (`forward_calls`), and the three
quantities are reported separately everywhere:

- `unique_sequences_encoded` — distinct proteins handed to the model
- `model_forward_batch_calls` — actual `model(tokens)` calls
- `cache_hits` — requests served from the store with no forward

For the LLR: `masked_positions`, `masked_inputs` and
`model_forward_batch_calls` are likewise separate.

### 5. Config can no longer drift from runtime

`configs/esm_repr_v1.yaml` is now both a runtime config and a validated one.
`src/embeddings/contract.py` checks 25 contract fields (model, backend, layers,
layer convention, precisions, window rule + version, split rule + version,
alignment version, slot kinds, array names, LLR method/version/AA order,
embedding dim, batching rules) against the Python constants at the start of
every CLI subcommand, before any model is loaded, and refuses to run while
naming each field that disagrees. The CLI also takes `paths`,
`batching.batch_size`, `window.W` and `esm.device` from the file, so the
documented values are the ones actually used.

### Fixture

`python -m src.embeddings.cli fixture --device cuda --batch-size 4` —
**66/66 checks passed** on the real `esm2_t33_650M_UR50D`. The fixture now
covers **both real duplication rows**, both termini, and the exact window
budgets:

| fixture | var_id | HGVSp | MUT len | slots | slot kinds |
|---|---|---|---|---|---|
| missense | chr17_58692701_T_A | p.Ser20Thr | 376 | 21 | paired 21 |
| synonymous | chr17_58692701_TCT_AGC | p.Ser20= | 376 | 21 | paired 21 |
| deletion | chr17_58692701_TCT_ | p.Ser20del | 375 | 21 | paired 20, wt_only 1 |
| duplication_186 | chr17_58696842__AAA | p.Lys186dup | 377 | 21 | paired 20, mut_only 1 |
| duplication_351 | chr17_58734138__TGT | p.Val351dup | 377 | 21 | paired 20, mut_only 1 |
| delins | chr17_58732531_TGTTTCAAATCA_ | p.Leu338_Lys342delinsGln | 372 | 26 | paired 20, wt_only 5, mut_only 1 |
| missense_nterm | chr17_58692647_C_G | p.Arg2Gly | 376 | 12 | paired 12 |
| missense_cterm | chr17_58734217_T_A | p.Leu376Ile | 376 | 11 | paired 11 |

Both duplications were checked for residue identity on every paired slot, for
the `+1` frame shift after the insertion boundary and for the split rule reading
only `{pp, pp+1}`: `p.Lys186dup` → `{train}` → in scope; `p.Val351dup` →
`{test, val}` → excluded.

Full suite: **143 passed** (`.venv/bin/python -m pytest tests/ -q`).

---

## Task C artifacts (regenerated)

Everything below was regenerated from the corrected code at commit
`8c2fc21` with a clean tree, in this order: audit → benchmark → export → LLR.
Every previous artifact was invalid: `window_rule_version`,
`alignment_version` and the new `split_rule` / `split_rule_version` are all
provenance-identity fields, so the old caches would be refused as stale anyway.

Commands:

```
.venv/bin/python -m src.embeddings.cli fixture --device cuda --batch-size 4
.venv/bin/python -m src.embeddings.variant_map --out-dir data/variant_audit
.venv/bin/python -m src.embeddings.cli benchmark --device cuda --batch-size 8 --n-seqs 32
.venv/bin/python -m src.embeddings.cli export --device cuda --batch-size 8
.venv/bin/python -m src.embeddings.cli llr --device cuda --batch-size 8 --diagnostic
.venv/bin/python -m pytest tests/ -q
```

### rows / sequences / forward calls

| | old (invalid) | regenerated |
|---|---|---|
| input rows | 5,887 | 5,887 |
| supported rows | 5,658 | **5,885** |
| failed rows | 229 (`cross_span_edit`) | **2 (`cross_span_edit`)** |
| rows by split | train 3,997 / val 829 / test 832 | **train 4,123 / val 881 / test 881** |
| rows by variant type | missense 4,555 / synonymous 973 / indel 130 | **missense 4,555 / synonymous 973 / indel 357** |
| rows by edit type | — | missense 4,555 / synonymous 973 / deletion 356 / duplication 1 |
| alignment width `A` | 24 | **21** |
| unique MUT proteins | 2,500 | **2,717** |
| `unique_sequences_encoded` | (reported as 2,500 "forwards") | **2,717** |
| `model_forward_batch_calls` | not measured | **342** |
| `cache_hits` | 2,862 | **3,079** |
| `sequence_requests` | — | 5,796 |
| stored MUT window tensors | 2,861 | **3,078** |

The old report's `actual_esm_forwards=2500` was a sequence count. The
regenerated run separates them: **2,717 distinct proteins** (including the WT)
were encoded in **342 actual `model(tokens)` calls** at batch size 8, and 3,079
of the 5,796 sequence requests were served from the in-memory store with no
forward at all.

Window widths across the export (`slots_per_row`): 5,596 rows get the full 21
slots; 289 rows near a terminus get 11–20. Nothing exceeds 21 — the 26-slot
delins is out of scope, which is why the fixture's `A` (26) still exceeds the
export's (21).

### frozen cache

`python -m src.embeddings.cli export --device cuda --batch-size 8` — 145.6 s
(forward 140.5 s, alignment 1.6 s, serialization 2.9 s; model load 10.4 s
reported separately).

| artifact | bytes |
|---|---|
| `data/esm_repr_v1/frozen_repr_v1.pt` | 1,003,276,698 |
| `data/esm_repr_v1/frozen_repr_v1.provenance.json` | 2,860 |
| `data/esm_repr_v1/export_report.json` | 1,564 |
| `data/esm_repr_v1/benchmark.json` | 3,162 |
| `data/esm_repr_v1/fixture_report.json` | 16,501 |
| `data/esm_repr_v1/fixture_repr.pt` (+ provenance) | 8,984,446 (+2,856) |
| directory total | 1,012,288,087 |

`wt_full [3,376,1280]` stored once, 3,078 deduplicated MUT window tensors
`[3,21,1280]`, per-row coordinate/mask arrays, and label-free identifiers.
`H_WT` and `delta_H` are reconstructed on load in FP32, bit-exact.

Provenance (`provenance_hash =
b3739cda25435959fcf42df8831e7bbb5d4e898ae77f066b4cad0892a4a6b377`) now also
records `split_rule` and `split_rule_version` alongside the window rule, and
both are provenance-identity fields. Recorded git state:
`8c2fc21edd96371719fcfdaf01d17f1f38eec840`, branch `minseon/esm-module`,
`dirty: false`. Verified after the run: reload matches the expected provenance,
`no targets stored = True`, `p.Lys186dup` present with 20 paired + 1 mut_only
slots, `p.Val351dup` and the delins absent, synonymous valid `delta_H` exactly
0.0.

### benchmark

`--n-seqs 32`, 32 distinct real sequences (375 ×3, 376 ×28, 377 ×1), warm-up
excluded, forward timed with CUDA events.

| | |
|---|---|
| GPU / torch / CUDA | Tesla V100-SXM2-32GB / 2.5.1+cu124 / 12.4 |
| precision (forward / cache / delta) | fp32 / fp32 / fp32, autocast off |
| batch size / layers / W / A | 8 / [31,32,33] / 10 / **21** |
| checkpoint + model load (separate) | 10.46 s, no download |
| tokenization / forward | 0.087 s / 1.528 s |
| alignment / serialization | 0.022 s / 0.022 s |
| throughput | **18.83 seq/s** overall, 20.94 forward-only |
| peak VRAM allocated / reserved | 2,915.4 MiB / 3,122.0 MiB |
| process RSS | 1,128.6 MiB |
| sample output | 30,967,571 B for `[32,3,21,1280]` |

Projection from these numbers: 1,489,346,618 B for the full export; actual
1,003,276,698 B — the projection still scales serialization per row rather than
per stored unique window, so it over-estimates.

### 650M LLR / profile

`--diagnostic`, 19.1 s of masked forwards.

| artifact | bytes |
|---|---|
| `data/esm650m_llr/llr_650m.csv` | 878,731 |
| `data/esm650m_llr/profile_650m.npz` | 29,170 |
| `data/esm650m_llr/provenance.json` | 1,914 |
| `data/esm650m_llr/llr_report.json` | 1,250 |
| `data/esm650m_llr/llr_diagnostic_trainval.json` | 237 |

| | |
|---|---|
| rows in scope | **5,885** (was 5,658) |
| `masked_positions` | 375 |
| `masked_inputs` | 375 |
| `model_forward_batch_calls` | **47** |
| profile shape | `[375, 20]`, AA order `ACDEFGHIKLMNPQRSTVWY` |
| LLR valid | 4,555, all missense |
| LLR undefined | 1,330 — synonymous 973, codon_deletion 349, clinical_inframe_deletion 7, clinical_inframe_insertion 1 |

The masked-WT profile is over WT positions only, so it is numerically identical
to the previous run; the scope change adds the 227 newly in-scope deletion rows
to the table with `llr_valid=False` and an explicit reason. No fabricated
`LLR=0`. `data/baseline_llr.csv` (HF `esm2_t30_150M_UR50D`) was not read, reused
or overwritten. Train/val-only diagnostic, test labels never opened:
Spearman(LLR, z) = **0.504** (train, n=3,195) and **0.611** (val, n=671) —
unchanged, as expected.

### audit

`data/variant_audit/audit.json` (11,715,145 B) and `validated_manifest.csv`
(4,518,459 B). The manifest now carries `left_boundary_wt`,
`right_boundary_wt`, `right_offset`, `edited_wt_positions`,
`split_positions_considered` and `split_rule_version` per row, so each scope
decision can be re-derived from the CSV alone.

### 실패하거나 지원되지 않은 항목

- **2 rows excluded**, both genuine `cross_span_edit`: `p.Leu338_Lys342delinsGln`
  (edited WT 338–342 spans train/val) and `p.Val351dup` (insertion boundary
  351/352 spans test/val). Both are still exercised by the fixture and the test
  suite, which is why the fixture's `A` is 26 and the export's is 21.
- No row failed for an ESM, alignment, window or serialization reason;
  `failed_reasons` is `{"cross_span_edit": 2}` and nothing else.
- `ins` (a pure HGVS insertion) still has no rows in this cohort. Its boundary
  handling was corrected here (it previously emitted no left boundary pair and
  placed the right pair one residue too far) and is covered by unit tests only.
- The benchmark's length mix is still dominated by 376 aa, because the cohort is.

---

## Task C

> **The numeric tables in this section are the ORIGINAL Task C run**, produced
> before the split-rule and window-definition defects were found. They are kept
> as the record of what was run then. The corrected numbers are in
> *Task C correction round* above and *Task C artifacts (regenerated)* below;
> where the two disagree, the corrected ones are authoritative.

### 바뀐 파일

Modified:

- [src/embeddings/esm_encoder.py](../src/embeddings/esm_encoder.py) — the equal-length
  batch assumption is now an explicit failure in `_forward()` instead of a silent
  truncation, plus `_resolve_layers()` and a new `encode_by_length()` that groups by
  residue length so indel MUT proteins never share a batch with the 376 aa WT.
- [src/embeddings/variant_map.py](../src/embeddings/variant_map.py) — new shared
  `_edit_geometry()` used by `_indel_span_info`, `_alignment_for_indel` and
  `build_variant_record`. This corrects `dup` and `ins`; see *Task B correction* below.
  The Task B schema, slot vocabulary, coordinate contract and all existing tests are
  unchanged.

Added:

- [src/embeddings/representation_cache.py](../src/embeddings/representation_cache.py)
- [src/embeddings/likelihood.py](../src/embeddings/likelihood.py)
- [src/embeddings/cli.py](../src/embeddings/cli.py) — `fixture` / `benchmark` / `export` / `llr`
- [tests/test_representation_cache.py](../tests/test_representation_cache.py) — 26 model-free tests
- [tests/test_esm_integration.py](../tests/test_esm_integration.py) — 23 real-checkpoint tests
- [configs/esm_repr_v1.yaml](../configs/esm_repr_v1.yaml)

Regenerated: `data/variant_audit/{audit.json,validated_manifest.csv}`. The
checked-out copies were stale — written by an older `variant_map.py` and reporting
`in_eval_scope=True` for all 5,887 rows with a `slot_kind_counts` block in the old
format. Rerunning the current code gives 5,658 supported / 229 excluded. The
originals were backed up to the session scratchpad before overwriting. `data/` is
gitignored, so no cache or audit artifact is committed.

Full suite: **104 passed** (`.venv/bin/python -m pytest tests/ -q`, 20.8 s).

### Task B correction (the one incompatibility found)

`_alignment_for_indel` gave `dup` no inserted residues, so the post-edit frame shift
was 0. For `p.Lys186dup` it paired **WT187 (H) with MUT187 (K)** — a subtraction
between two different residues — and emitted no `mut_only` slot for the duplicated
copy. `ins` had a matching off-by-one (`mut_only` at `start..start+k-1` instead of
`start+1..start+k`; 0 rows affected, fixed for consistency). Both now derive from
`_edit_geometry()`, which mirrors `dataset.build_mutant_sequence()` exactly.
`p.Lys186dup` now yields `paired 185→185, paired 186→186, mut_only MUT187, paired
187→188`, and every paired slot matches the same residue identity on both sides.
`del`/`delins` were already correct and are byte-for-byte unchanged.

### fixture correctness 결과

`python -m src.embeddings.cli fixture --device cuda --batch-size 4` — **46/46 checks
passed**, real `esm2_t33_650M_UR50D`, FP32, `model.eval()`, `no_grad`.
Report: `data/esm_repr_v1/fixture_report.json`.

| fixture | var_id | HGVSp | MUT len | slots | slot kinds |
|---|---|---|---|---|---|
| missense | chr17_58692701_T_A | p.Ser20Thr | 376 | 21 | paired 21 |
| synonymous | chr17_58692701_TCT_AGC | p.Ser20= | 376 | 21 | paired 21 |
| deletion | chr17_58692701_TCT_ | p.Ser20del | 375 | 23 | paired 22, wt_only 1 |
| duplication | chr17_58696842__AAA | p.Lys186dup | 377 | 24 | paired 23, mut_only 1 |
| delins | chr17_58732531_TGTTTCAAATCA_ | p.Leu338_Lys342delinsGln | 372 | 28 | paired 22, wt_only 5, mut_only 1 |

Verified:

- layers 31/32/33 present, each `[B, 376, 1280]`, and mutually distinct
- individual vs batched forward: **max abs 7.019e-04, max relative 1.911e-06**, tol 1e-05.
  The tolerance is *relative* on purpose: layers 31/32 are pre-final-LayerNorm and
  reach |h| ≈ 490, so the absolute figure is FP32 GEMM reduction-order drift from the
  batch shape. The same drift appears on CPU (2–3e-04), and two identical sequences at
  different positions of one batch agree **bit-exactly** — so this measures float
  associativity, not a batching bug.
- mixed-length batch raises instead of truncating to the first sequence's length
- synonymous reuses the same cached WT tensor object; valid `delta_H` is **exactly 0.0**
  over 21 valid slots, and `H_WT == H_MUT` bit-exact
- delins: `WT343 → MUT339` and `WT337 → MUT337` preserved; WT 338–342 are `wt_only`;
  the inserted Q is one `mut_only` slot at MUT338 (residue confirmed `Q`)
- delins: no `wt_only`/`mut_only` slot is `delta_valid`; `delta_H` is exactly 0 there;
  `wt_only` keeps a real WT representation with the MUT side zeroed, and vice versa
- no same-array-index subtraction: 11 shifted paired slots, and the shifted delta was
  checked bit-exact against `MUT[mut_pos] - WT[wt_pos]`
- `[B,3,A,1280]` contract holds (`(5, 3, 28, 1280)`)
- save → reload: tensors, masks, provenance and H_WT/H_MUT/delta_H all bit-identical
- stale-provenance rejection fires on `base_checkpoint_hash`, `repr_layers`,
  `window_rule_version`, `cache_precision`, `alignment_version`, `manifest_hash`

Also verified separately against the **production 1 GB cache**: matching provenance
loads; 12 mutated identity fields (`base_checkpoint_hash`, `adapter`,
`forward_precision`, `cache_precision`, `repr_layers`, `window_W`,
`window_rule_version`, `alignment_version`, `variant_map_hash`, `manifest_hash`,
`split_schema_hash`, `sequence_set_hash`) are each refused with `StaleCacheError`.

### benchmark 실제 측정값

`python -m src.embeddings.cli benchmark --device cuda --batch-size 8 --n-seqs 32` →
`data/esm_repr_v1/benchmark.json`. 32 distinct real sequences spanning all three real
lengths (375 ×1, 376 ×30, 377 ×1). Warm-up run first, then
`torch.cuda.reset_peak_memory_stats()`; forward timed with CUDA events + synchronize.

| | |
|---|---|
| GPU / VRAM | Tesla V100-SXM2-32GB / 34,072,559,616 B (32,494 MiB) |
| torch / CUDA runtime / cuDNN / driver | 2.5.1+cu124 / 12.4 / 90100 / 580.173.02 |
| precision (forward / cache / delta) | fp32 / fp32 / fp32, autocast off |
| batch size / repr layers / window / A | 8 / [31,32,33] / ±10 / 24 |
| sequences (total / unique / lengths) | 32 / 32 / 375–377 |
| checkpoint load + model load | 10.581 s (checkpoint already cached; **no download**) |
| tokenization | 0.090 s |
| forward | 1.557 s (CUDA events) |
| alignment / window | 0.098 s |
| serialization / write | 0.026 s |
| overall throughput | **17.65 seq/s** (20.55 seq/s forward-only; 0.0567 s/seq) |
| alignment throughput | 327.2 rows/s |
| `torch.cuda.max_memory_allocated` | 3,056,954,368 B (2,915.3 MiB) |
| `torch.cuda.max_memory_reserved` | 3,248,488,448 B (3,098.0 MiB) |
| process RSS | 1,191,714,816 B (1,136.5 MiB) |
| sample output bytes | 35,391,251 B for `[32,3,24,1280]` H_WT+H_MUT+delta_H |

Setup time (checkpoint + model load, 10.58 s) is reported separately from processing
time and is not mixed into any throughput figure.

Projection from these measurements → **163.7 s (2.73 min)** and ~1.58 GB for the full
export. Actual: **131.9 s and 1.065 GB** — the projection over-estimated because it
scaled serialization per row rather than per stored unique window.

### frozen cache 산출물 / 크기

`python -m src.embeddings.cli export --device cuda --batch-size 8` (2 m 29 s wall).

| artifact | bytes |
|---|---|
| `data/esm_repr_v1/frozen_repr_v1.pt` | 1,065,411,098 |
| `data/esm_repr_v1/frozen_repr_v1.provenance.json` | 2,157 |
| `data/esm_repr_v1/export_report.json` | 1,024 |
| `data/esm_repr_v1/benchmark.json` | 3,134 |
| `data/esm_repr_v1/fixture_report.json` | 9,712 |
| `data/esm_repr_v1/fixture_repr.pt` (+ provenance) | 7,936,702 (+2,154) |
| directory total | 1.0 GB |

Contents: `wt_full` `[3,376,1280]` stored once; 2,861 deduplicated MUT window tensors
`[3,24,1280]`; per-row `wt_pos`/`mut_pos`/`wt_present`/`mut_present`/`delta_valid`/
`token_valid`/`slot_kind`; `var_id`/`split`/`variant_type`/`edit_type`/`pp`/sequence
hashes. `H_WT` and `delta_H` are reconstructed on load in FP32 (bit-exact — cache
precision is FP32). Full MUT hidden states are **not** persisted, only windowed slices.

Provenance recorded: backend, model name, base checkpoint hash, `adapter=none` /
`adapter_state=frozen`, repr layers, layer convention, forward precision, cache
precision, delta precision, device, WT sequence hash, sequence-set hash, manifest
path + hash, split-schema hash, alignment version, variant-map hash, window rule +
version + W, per-row sequence hashes, code hashes for 4 files, git commit/branch/dirty,
`contains_targets=false`, `target_columns_excluded`. Provenance hash:
`e28df4905cf29bcc5090543056fc2d61da5b747a385f34ae5ba0a532ea88f741`.

**No SGE target or functional classification is stored in the cache** — asserted by
both the fixture and the test suite.

### 650M LLR/profile 산출물 / 크기

`python -m src.embeddings.cli llr --device cuda --batch-size 8 --diagnostic` (36 s).

| artifact | bytes |
|---|---|
| `data/esm650m_llr/llr_650m.csv` | 848,678 |
| `data/esm650m_llr/profile_650m.npz` | 29,171 |
| `data/esm650m_llr/provenance.json` | 1,861 |
| `data/esm650m_llr/llr_report.json` | 1,206 |
| `data/esm650m_llr/llr_diagnostic_trainval.json` | 237 |
| directory total | 876 KB |

- scoring method `masked_marginal_wt`, from the **same repo-native 650M checkpoint** as
  the representations; logits and `log_softmax` in FP32
- 375 unique missense positions, **375 masked forwards** for 4,555 missense rows
  (one mask per position, every substitution at that position read off it)
- profile `[375, 20]`, AA order `ACDEFGHIKLMNPQRSTVWY` stored with the data
- per row: `pp`, `wt_aa`, `ref_aa`, `mut_aa`, `logp_wt`, `logp_mut`, `llr`, `llr_valid`,
  `llr_undefined_reason`, `scoring_method`, `aa_order`, `ref_matches_wt`
- `ref_aa` matched the WT residue for all 4,555 valid rows
- **LLR valid for missense only: 4,555.** 1,103 non-missense rows carry `llr=NaN`,
  `llr_valid=False` and an explicit reason (synonymous 973, codon_deletion 126,
  clinical_inframe_deletion 3, clinical_inframe_insertion 1). No fabricated `LLR=0`.
- `data/baseline_llr.csv` (HF `facebook/esm2_t30_150M_UR50D`) was **not** read, reused
  or overwritten; it is untouched on disk. Confirmed the new numbers are a genuinely
  different computation: over the 4,555 shared rows, Spearman(650M, 150M) = 0.835 and
  mean |difference| = 2.337.
- Development diagnostic, **train/val only, test labels never opened**:
  Spearman(LLR, z) = 0.504 (train, n=3,195) and 0.611 (val, n=671).

### rows / unique sequences / actual forwards

| | |
|---|---|
| input rows | 5,887 |
| supported rows (`in_eval_scope=True`) | 5,658 |
| successful rows | 5,658 |
| failed rows | 229 |
| rows by split | train 3,997 / val 829 / test 832 |
| rows by variant type | missense 4,555 / synonymous 973 / inframe_indel 130 |
| unique WT sequences | 1 |
| unique MUT sequences | 2,500 (the WT protein is one of them — every synonymous row) |
| **actual ESM forwards** | **2,500** |
| forward cache hits | 2,862 |
| forwards avoided vs one-per-row | 3,159 (5,659 → 2,500) |
| stored MUT window tensors | 2,861 (= 2,500 sequences × distinct window layouts) |
| masked LLR forwards | 375 |
| alignment width A | 24 (fixture 28, because of the 5-residue delins) |

DNA rows (5,658) and unique protein sequences (2,500) are recorded separately in
`export_report.json` under `dna_rows_vs_unique_protein`.

### 실패하거나 지원되지 않은 항목

- **229 rows excluded, all `cross_span_edit`** (Task B's rule: an indel whose window
  spans more than one split). By edit type: deletion 227, delins 1, duplication 1.
  These are excluded by Task B, not by Task C, and no Task C code path failed on them.
- The delins `p.Leu338_Lys342delinsGln` is one of those 229, so it is **not** in the
  full export. It is still exercised in the fixture and in the test suite, which is why
  the fixture's A (28) exceeds the export's A (24).
- No row failed for an ESM, alignment, or serialization reason: `failed_reasons` is
  `{"cross_span_edit": 229}` and nothing else.
- `ins` (a pure HGVS insertion) has no rows in this cohort; its corrected geometry is
  covered by unit tests only, not by real data.
- The benchmark's length mix is dominated by 376 aa (30 of 32) because the cohort is —
  only 126 unique sequences are 375 aa and exactly 1 is 377 aa. One of each is included
  so the mixed-length path is exercised, but the timing is effectively a 376 aa figure.

### 아직 안 된 것

- Q/LoRA adaptation, Bradley-Terry training, Mixout, KL, stage 1/2 training — all
  deliberately out of Task C scope, none started.
- No window / layer / rank search. `W=10` and layers `[31,32,33]` are fixed by contract.
- No test-label performance exploration. Test rows have features and a label-free LLR,
  but `z_score_D4_D14` / `functional_classification` for test rows were never opened.
  The only metric reported anywhere is the train/val LLR diagnostic above.
- The legacy M1–M4 pooling results are untouched and are **not** relabelled as F-seq
  results.
- The frozen cache is not yet wired into `train.py` / `dataset.py`; nothing downstream
  consumes `frozen_repr_v1.pt` yet.

### 다음 작업

1. Wire `RepresentationCache` into the data path so a training run can consume
   `H_WT`/`H_MUT`/`delta_H` with the masks, gated on the provenance hash.
2. Decide the readout over the `A=24` slots (`token_valid` and `slot_kind` are the
   inputs to that decision) before any layer or window sweep.
3. Only then open the adaptation track (LoRA on `q_proj`/`v_proj` per CLAUDE.md §2),
   keeping the frozen cache as the fixed baseline.

---

## Commands executed (Task C, original run)

1. `df -h ~` — storage gate check
2. `.venv/bin/python -m pytest tests/test_variant_map.py -q`
3. `.venv/bin/python -m pytest tests/test_representation_cache.py -q`
4. `.venv/bin/python -m src.embeddings.cli fixture --device cuda --batch-size 4`
5. `.venv/bin/python -m src.embeddings.cli benchmark --device cuda --batch-size 8 --n-seqs 32`
6. `.venv/bin/python -m src.embeddings.cli export --device cuda --batch-size 8`
7. `.venv/bin/python -m src.embeddings.cli llr --device cuda --batch-size 8 --diagnostic`
8. `.venv/bin/python -m src.embeddings.variant_map --out-dir data/variant_audit`
9. `.venv/bin/python -m pytest tests/ -q`

---

## Task A / Task B record

### Task A

- Environment verified in `/home/coder/deepCANCERsite/.venv` with no package changes:
  Python 3.10.12, PyTorch 2.5.1+cu124, CUDA 12.4, Tesla V100-SXM2-32GB (32,494 MiB).
- Repo-native ESM import confirmed: `esm.__file__ = /home/coder/deepCANCERsite/esm/__init__.py`.
- See [docs/ESM_AGENT_CONTEXT.md](ESM_AGENT_CONTEXT.md).

### Task B

- Canonical row-level + alignment-slot schema in
  [src/embeddings/variant_map.py](../src/embeddings/variant_map.py)
- Per-slot coordinate contract uses integer 0 for absent positions, with
  `wt_present` / `mut_present` controlling actual occupancy
- `supported_by_split` and `supported_by_split_and_variant_type` in the audit output
- Real fixture validation for `p.Leu338_Lys342delinsGln` and manifest-level cross-span
  exclusion audit
- **The cross-span rule was corrected** — it read the alignment (paired context
  residues and MUT coordinates) instead of the directly edited WT span. See
  *Task C correction round*. `split_rule_v2_edited_wt_span_only`.
- Regression checks in [tests/test_variant_map.py](../tests/test_variant_map.py),
  extended with the split-rule cases (flank-only difference, multi-residue delins,
  both real duplication rows, whole-manifest outcome)
- `data/variant_audit/{validated_manifest.csv,audit.json}` regenerated from the
  corrected rule: **5,885 supported / 2 excluded**
