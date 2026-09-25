# RAD51C hidden representations and adapter checkpoints

Frozen remains the reference. This handoff includes the completed matched Q and Q+Mixout experiment: folds 0/1, adapter seeds 101/102, all eight candidates. These are exploratory exports from reused development folds, not adopted replacements or full-data fits. See experiment/RESULTS.md.

## Files and download

The Git-tracked manifest, loader, metadata, fold assignments and receipts accompany nine representation caches and eight pass-5 LoRA checkpoints. Binary files belong in payload/ and are distributed separately as GitHub release assets. Publication is pending until an upload receipt exists. Download every `.pt` release asset into this directory's `payload/` folder, retaining its filename.

The caches use deduplicated storage. `load_hr.py` reconstructs H_WT, H_MUT and delta_H exactly in FP32; it does not rerun ESM. This saves several GB compared with storing three expanded tensors per variant. Install Python 3.10+ and PyTorch. No GPU or pretrained ESM checkpoint is required for consuming the features.

```bash
python load_hr.py  # verify all distributed files against manifest.json
```

```python
from load_hr import HR
features = HR('fold0_q_mixout_101', fold=0)
ids = features.ids('train')[:32]
batch = features.batch(ids)
# batch['H_WT'], batch['H_MUT'], batch['delta_H']: [B, 3, 21, 1280]
# Compare against HR('frozen', fold=0), using exactly the same IDs and roles.
```

Join by `var_id`, never by row number. Layers are 31/32/33. Use token_valid to mask padding; indel gaps are real slots. delta_valid marks paired subtraction. Synonymous delta is exactly zero. Coordinates are 1-based with zero for an absent side. Preserve the accompanying occupancy masks, slot_kind_vocab and edit metadata.

## Splits and evaluation

Use fold0_roles.json or fold1_roles.json, matching the adapter. The loader returns these roles; the legacy split embedded in the original cache is not the fold assignment. Outer fold i is test, fold (i+1) mod 7 is validation, and the other five folds train. Role ID hashes have been checked against every adapter's locked training identity. Do not use an adapter trained on your evaluation labels. Labels are not included here; join your own assay labels by ID and fit any target scaler on training IDs only. Existing folds have already been examined and remain exploratory.

Every cache covers the same 5,885 supported variants, including train/val/test features. This does not mean every row was used for adapter training. `provenance/` records the exact training IDs and checkpoint identity. All seeds are included; no candidate was selected for transfer based on favorable results.

## Checkpoints

`*_adapter.pt` contains LoRA factors, not the full ESM encoder or our prediction head. Reconstructing an encoder requires the exact pretrained esm2_t33_650M_UR50D base and compatible training source. The base SHA-256 and Q configuration are in the adapter provenance. Attention Q and V in blocks 30–32 use rank 8, alpha 16. These checkpoints are provided for reproducibility; use the cached features to train your downstream model directly.

Local payload files are hard links to completed immutable experiment artifacts to save disk space. Do not modify them in place. Source artifact paths in manifest.json are audit references, not dependencies on the receiver's machine.

## Manual GitHub release upload

1. Push the handoff documentation commit when repository authentication is available.
2. In the repository's Releases page, create a release with a new tag such as `module-a-hr-v1` targeting that commit. Mark it as a pre-release because the adapted candidates remain exploratory.
3. Attach `data/module_a_hr_v1_support.zip` and all 17 `.pt` files inside this folder's `payload/` directory. The support ZIP contains the loader, metadata, manifest, instructions and verification receipt; it excludes the large tensors.
4. Publish the release and send your teammate the release URL. They extract the support ZIP, put the `.pt` assets in `module_a_hr_v1/payload/`, then run `python load_hr.py` from `module_a_hr_v1/`.

GitHub rejects ordinary Git files above 100 MiB; each representation cache here is approximately 957 MiB. Use release assets for these binaries, or separately configure Git LFS. Do not force-add payload/ into normal Git history. The ordinary Git commit alone does not deliver the tensors.
