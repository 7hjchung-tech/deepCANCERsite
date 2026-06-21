# src/analysis — Layer Representation Convergence Diagnostics

## Purpose

This module tracks how each ESM-2 transformer layer's per-site MUT
representation converges toward the final layer (layer 33) representation as
LoRA fine-tuning progresses.

**This is a diagnostic tool, not a quality metric.**
A decreasing layer-distance does not mean better fine-tuning.  The curves
produced here should be plotted *alongside* task performance (e.g. LFC
validation Spearman correlation) to detect phenomena such as:

- Layer collapse: distances shrink but effective rank collapses simultaneously
- Over-adaptation: distances spike then collapse at a specific layer
- Stable intermediate layers: early layers converge faster than middle ones

## Reproducibility definitions

| Symbol | Definition |
|--------|-----------|
| `h_l`  | `window_pool(H_MUT_l, pos, W)` — window-pooled MUT hidden state at layer `l` |
| `h_norm` | `h_33` — same pooling at the final transformer layer (per variant) |
| `Sigma_l` | Ledoit-Wolf estimate of the covariance of all per-residue representations at layer `l`, computed **once** on the pretrained (LoRA B=0) baseline |

Because `Sigma_l` is frozen at the pretrained baseline (`freeze_at="pretrained"`),
the Mahalanobis metric space is fixed across epochs, making epoch-to-epoch distance
comparisons valid.

Setting `freeze_at="per_epoch"` re-estimates `Sigma_l` each epoch; this makes
cross-epoch absolute comparisons unreliable (the ruler changes length), so only
use it to inspect per-epoch geometry, not trends.

## Output format

`runs/layer_analysis/layer_metrics.jsonl` (or `.csv`) — one record per
`(epoch, layer_index)` pair, 33 records per snapshot:

```
epoch  layer_index  maha_dist  mean_norm  variance  eff_rank
```

Append one `snapshot()` call per epoch; plot `layer_index` on the x-axis and
`maha_dist` on the y-axis, with each epoch as a separate curve.

## Usage

```python
from src.analysis.lens_tracker import LensTracker

tracker = LensTracker(encoder, reference_variants, cfg["analysis"])

# Inside training loop, after optimizer.step():
records = tracker.snapshot(model, epoch=epoch)
```
