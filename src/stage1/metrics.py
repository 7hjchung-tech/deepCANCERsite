"""Evaluation metrics, with undefined results reported as undefined.

Spearman is undefined when there is nothing to rank -- too few rows, a constant
prediction (a model that has collapsed), a constant target, or a non-finite
value. In every one of those cases `spearman()` returns `value=None` and a
`undefined_reason`, never a NaN that a caller could compare against a real
score and silently pick as "best".

MAE / RMSE / MSE are always reported in RAW z-score units, i.e. after the
target scaler's inverse transform. Spearman is rank-based, so it is invariant
to that affine map and agrees in either space; `spearman_agreement()` exists to
demonstrate that rather than assume it.

The rank convention (average ranks, ties shared) matches scipy.stats.rankdata
and the legacy train.py implementation; a test asserts agreement with scipy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np

#: Fewer rows than this and a rank correlation is meaningless.
MIN_SPEARMAN_ROWS = 3


def rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks with ties shared -- scipy.stats.rankdata's convention."""
    a = np.asarray(a, dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    ranks[order] = np.arange(1, len(a) + 1, dtype=np.float64)
    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return ranks


@dataclass(frozen=True)
class SpearmanResult:
    """A Spearman value that knows whether it exists."""

    value: Optional[float]
    n: int
    undefined_reason: Optional[str] = None

    @property
    def is_defined(self) -> bool:
        return self.value is not None

    def to_dict(self) -> Dict[str, object]:
        return {"value": self.value, "n": self.n, "undefined_reason": self.undefined_reason}

    def __float__(self) -> float:
        if self.value is None:
            raise ValueError(
                f"Spearman is undefined ({self.undefined_reason}); it has no float value"
            )
        return float(self.value)


def spearman(y_true: Sequence[float], y_pred: Sequence[float]) -> SpearmanResult:
    t = np.asarray(y_true, dtype=np.float64).ravel()
    p = np.asarray(y_pred, dtype=np.float64).ravel()
    if t.shape != p.shape:
        raise ValueError(f"shape mismatch: y_true {t.shape} vs y_pred {p.shape}")

    n = int(t.size)
    if n < MIN_SPEARMAN_ROWS:
        return SpearmanResult(None, n, f"too_few_rows(n={n}<{MIN_SPEARMAN_ROWS})")
    if not np.isfinite(t).all():
        return SpearmanResult(None, n, "non_finite_targets")
    if not np.isfinite(p).all():
        return SpearmanResult(None, n, "non_finite_predictions")
    if float(t.std()) == 0.0:
        return SpearmanResult(None, n, "constant_targets")
    if float(p.std()) == 0.0:
        return SpearmanResult(None, n, "constant_predictions")

    rt, rp = rankdata(t), rankdata(p)
    if float(rt.std()) == 0.0 or float(rp.std()) == 0.0:
        return SpearmanResult(None, n, "constant_ranks")
    value = float(np.corrcoef(rt, rp)[0, 1])
    if not np.isfinite(value):
        return SpearmanResult(None, n, "non_finite_correlation")
    return SpearmanResult(value, n, None)


def spearman_agreement(
    y_true_raw: Sequence[float],
    y_pred_raw: Sequence[float],
    y_true_std: Sequence[float],
    y_pred_std: Sequence[float],
) -> float:
    """|rho_raw - rho_std|. Should be ~0: the scaler is monotonic."""
    a, b = spearman(y_true_raw, y_pred_raw), spearman(y_true_std, y_pred_std)
    if not (a.is_defined and b.is_defined):
        return float("nan")
    return abs(float(a) - float(b))


def regression_metrics(y_true_raw: Sequence[float], y_pred_raw: Sequence[float]) -> Dict[str, float]:
    """MAE / RMSE / conventional MSE in RAW z-score units."""
    t = np.asarray(y_true_raw, dtype=np.float64).ravel()
    p = np.asarray(y_pred_raw, dtype=np.float64).ravel()
    if t.shape != p.shape:
        raise ValueError(f"shape mismatch: y_true {t.shape} vs y_pred {p.shape}")
    err = p - t
    return {
        "n": int(t.size),
        "raw_mae": float(np.mean(np.abs(err))),
        "raw_rmse": float(np.sqrt(np.mean(err ** 2))),
        "raw_mse": float(np.mean(err ** 2)),
    }


def evaluate_split(
    *,
    var_id: Sequence[str],
    variant_type: Sequence[str],
    y_true_raw: Sequence[float],
    y_pred_raw: Sequence[float],
    y_true_std: Sequence[float],
    y_pred_std: Sequence[float],
    optimization_loss: Optional[float] = None,
    objective: Optional[str] = None,
) -> Dict[str, object]:
    """The full validation report for one split.

    `missense_spearman` is the primary checkpoint-selection criterion. Indel and
    synonymous numbers are reported descriptively and are NOT selection inputs.
    """
    vt = np.asarray(list(variant_type))
    out: Dict[str, object] = {
        "n": int(len(list(var_id))),
        "objective": objective,
        "optimization_loss_standardized": optimization_loss,
        "missense_spearman": spearman(
            np.asarray(y_true_raw)[vt == "missense"],
            np.asarray(y_pred_raw)[vt == "missense"],
        ).to_dict(),
        "overall_spearman": spearman(y_true_raw, y_pred_raw).to_dict(),
        "spearman_raw_vs_standardized_absdiff": spearman_agreement(
            y_true_raw, y_pred_raw, y_true_std, y_pred_std
        ),
    }
    out.update(regression_metrics(y_true_raw, y_pred_raw))

    descriptive: Dict[str, object] = {}
    for kind in sorted(set(vt.tolist())):
        m = vt == kind
        descriptive[kind] = {
            "spearman": spearman(
                np.asarray(y_true_raw)[m], np.asarray(y_pred_raw)[m]
            ).to_dict(),
            **regression_metrics(np.asarray(y_true_raw)[m], np.asarray(y_pred_raw)[m]),
        }
    out["by_variant_type_descriptive"] = descriptive
    return out


def selection_score(report: Dict[str, object]) -> Optional[float]:
    """The checkpoint-selection criterion: validation missense Spearman.

    Returns None when it is undefined, so a caller must decide explicitly what
    to do rather than inheriting a NaN comparison that is always False.
    """
    return report["missense_spearman"]["value"]  # type: ignore[index]
