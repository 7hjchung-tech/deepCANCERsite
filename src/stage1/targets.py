"""Train-only target standardisation, and its exact inverse.

WHY
---
`z_score_D4_D14` spans roughly [-33.6, +4.9] with a bulk near 0, so a squared
error on the raw target is dominated by a handful of extreme depleted variants.
Optimisation therefore happens in STANDARDISED target space:

    mu_train    = mean(y_train)
    sigma_train = std(y_train)                 # population std, ddof = 0
    y_std       = (y - mu_train) / sigma_train

THE RULE THAT MATTERS
---------------------
`mu_train` and `sigma_train` are fitted on TRAINING ROWS ONLY, once per run,
and validation/test reuse those exact numbers. They are never refitted on a
non-train split -- doing so would leak the evaluation split's location and
scale into the numbers used to judge the model.

Every reported MAE / RMSE / MSE is computed AFTER `inverse_transform`, i.e. in
original z-score units. Spearman is rank-based and therefore invariant to this
affine map, so it agrees in either space up to floating-point noise.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Sequence

import numpy as np

#: A standard deviation at or below this is treated as degenerate rather than
#: silently replaced by 1.0 -- a constant target is a data problem, not a
#: scaling problem, and quietly dividing by 1.0 would hide it.
MIN_STD = 1e-8

#: Population standard deviation. Matches numpy's default and the convention
#: already used by the legacy train.py, so the two are directly comparable.
DDOF = 0

DEFAULT_TARGET_COLUMN = "z_score_D4_D14"


@dataclass(frozen=True)
class TargetScaler:
    """An immutable, fully-recorded affine target transform."""

    mean: float
    std: float
    ddof: int
    target_column: str
    n_train_rows: int
    fit_split: str
    std_convention: str

    # ------------------------------------------------------------------
    @classmethod
    def fit_on_train(
        cls,
        y_train: Sequence[float] | np.ndarray,
        *,
        target_column: str = DEFAULT_TARGET_COLUMN,
        fit_split: str = "train",
        ddof: int = DDOF,
    ) -> "TargetScaler":
        """Fit on training rows only. Raises on anything degenerate."""
        y = np.asarray(y_train, dtype=np.float64).ravel()
        if y.size < 2:
            raise ValueError(
                f"target scaler needs at least 2 {fit_split} rows, got {y.size}"
            )
        if not np.isfinite(y).all():
            n_bad = int((~np.isfinite(y)).sum())
            raise ValueError(
                f"{n_bad} non-finite target values in the {fit_split} split; "
                f"refusing to fit a scaler on them"
            )

        mean = float(y.mean())
        std = float(y.std(ddof=ddof))
        if not np.isfinite(std):
            raise ValueError(f"{fit_split} target std is not finite ({std})")
        if std <= MIN_STD:
            raise ValueError(
                f"{fit_split} target std is {std:.3e} <= {MIN_STD:.0e}: the target is "
                f"constant on this split, so standardisation is undefined. Fix the "
                f"cohort rather than substituting a fallback scale."
            )

        return cls(
            mean=mean,
            std=std,
            ddof=int(ddof),
            target_column=str(target_column),
            n_train_rows=int(y.size),
            fit_split=str(fit_split),
            std_convention=f"numpy population std, ddof={int(ddof)}",
        )

    # ------------------------------------------------------------------
    def transform(self, y: Sequence[float] | np.ndarray) -> np.ndarray:
        """raw -> standardised, using the fitted train statistics."""
        return (np.asarray(y, dtype=np.float64) - self.mean) / self.std

    def inverse_transform(self, y_std: Sequence[float] | np.ndarray) -> np.ndarray:
        """standardised -> raw z-score units. Exact inverse of transform()."""
        return np.asarray(y_std, dtype=np.float64) * self.std + self.mean

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TargetScaler":
        return cls(**d)

    def describe(self) -> str:
        return (
            f"TargetScaler({self.target_column}) fitted on {self.n_train_rows} "
            f"{self.fit_split} rows: mean={self.mean:.6f} std={self.std:.6f} "
            f"({self.std_convention})"
        )
