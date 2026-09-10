"""The two candidate Stage 1 objectives, and the metric they are not.

OPTIMISATION LOSSES -- both computed in STANDARDISED target space
-----------------------------------------------------------------
  huber      SmoothL1 / Huber with delta = 1.0. `delta` is in standardised
             target units, so "1.0" means one train standard deviation.
  half_mse   0.5 * mean((pred_std - target_std)^2).

The 0.5 is part of the OPTIMISATION LOSS ONLY. It halves the gradient scale
relative to plain MSE; it is not a different error measure and must never be
reported as an MSE.

EVALUATION METRIC -- computed in RAW z-score space
--------------------------------------------------
  conventional_mse   mean((pred_raw - target_raw)^2)

So for the same predictions there are three distinct numbers, and mixing them
up would make the two objectives look different when they are not:

    half_mse(std)          = 0.5 * mse(std)
    conventional_mse(raw)  = std_scale^2 * mse(std)

Nothing else belongs here. No MAE objective, no log-cosh, no ranking or
weighted or focal loss, and `delta` is fixed at 1.0 -- it is not a search space.
"""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np
import torch
import torch.nn as nn

#: Fixed by contract. Not a hyperparameter to search in this task.
HUBER_DELTA = 1.0


def huber_loss(
    pred_std: torch.Tensor, target_std: torch.Tensor, delta: float = HUBER_DELTA
) -> torch.Tensor:
    """Huber / SmoothL1 with delta=1.0, on standardised targets.

    Identical to `nn.HuberLoss(delta=1.0)`, which is what the legacy training
    loop already uses, so the two are directly comparable.
    """
    return nn.functional.huber_loss(pred_std, target_std, reduction="mean", delta=delta)


def half_mse_loss(pred_std: torch.Tensor, target_std: torch.Tensor) -> torch.Tensor:
    """0.5 * mean squared error on standardised targets. An OPTIMISATION loss.

    Not an MSE metric. Use `conventional_mse` for reporting.
    """
    return 0.5 * torch.mean((pred_std - target_std) ** 2)


#: The only two objectives Task D compares. Keys are the run labels.
OBJECTIVES: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "huber": huber_loss,
    "half_mse": half_mse_loss,
}

OBJECTIVE_DESCRIPTIONS: Dict[str, str] = {
    "huber": f"SmoothL1/Huber, delta={HUBER_DELTA} in standardised target units",
    "half_mse": "0.5 * mean((pred_std - target_std)^2), standardised target units",
}


def get_objective(name: str) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    if name not in OBJECTIVES:
        raise KeyError(
            f"unknown objective {name!r}; Task D compares exactly {sorted(OBJECTIVES)}"
        )
    return OBJECTIVES[name]


def conventional_mse(pred_raw: np.ndarray, target_raw: np.ndarray) -> float:
    """mean((pred_raw - target_raw)^2) in original z-score units. A METRIC.

    Deliberately not `half_mse_loss`: no 0.5, and raw units, not standardised.
    """
    p = np.asarray(pred_raw, dtype=np.float64)
    t = np.asarray(target_raw, dtype=np.float64)
    return float(np.mean((p - t) ** 2))
