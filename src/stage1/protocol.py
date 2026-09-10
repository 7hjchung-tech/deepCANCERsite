"""The FIXED PILOT protocol for the Huber vs 0.5*MSE comparison.

STATUS: specified, NOT exercised. The team's real token-level Stage 1 does not
exist in this repository, so no comparison has been run and no loss has been
selected. These constants exist so that, when Stage 1 arrives, both objectives
can be run under provably identical conditions without anyone re-deciding the
protocol after seeing a result.

WHERE THESE NUMBERS COME FROM
-----------------------------
Every value that the legacy M1-M4 pipeline already established is reused
verbatim from configs/base.yaml, so this is not a new recipe:

    optimizer            AdamW                      (train.py)
    learning rate        1e-4                       (base.yaml lr.head)
    weight decay         0.01                       (base.yaml)
    batch size           32                         (base.yaml)
    gradient clipping    global norm 1.0            (train.py)
    early-stop patience  10 epochs                  (train.py --patience)
    precision            fp32                       (base.yaml)
    seeds                42, 43, 44                 (base.yaml)

The remaining values (max_epochs, scheduler) had no established setting, so one
minimal choice is fixed here and labelled a pilot. NO hyperparameter search is
performed in Task D, and neither objective may be tuned separately from the
other: the only intentional difference between the two runs is `objective`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .losses import HUBER_DELTA, OBJECTIVE_DESCRIPTIONS
from .targets import DDOF, DEFAULT_TARGET_COLUMN


@dataclass(frozen=True)
class PilotProtocol:
    """Everything that must be identical between the two objective runs."""

    # --- optimisation (reused from configs/base.yaml where it existed) ---
    optimizer: str = "AdamW"
    learning_rate: float = 1.0e-4
    weight_decay: float = 0.01
    batch_size: int = 32
    gradient_accumulation: int = 1
    grad_clip_norm: float = 1.0
    precision: str = "fp32"
    scheduler: Optional[str] = None          # none established; none introduced
    max_epochs: int = 50                     # pilot value
    early_stopping_patience: int = 10
    seeds: Tuple[int, ...] = (42, 43, 44)
    init_policy: str = "torch default init, seeded per run before model construction"

    # --- data / target contract -----------------------------------------
    cohort: str = "in_eval_scope=True, verified against variant_map.build_variant_record"
    split_source: str = "data/split_manifest.csv split column (shipped assignment)"
    target_column: str = DEFAULT_TARGET_COLUMN
    scaler_fit_split: str = "train"
    scaler_std_ddof: int = DDOF
    optimisation_space: str = "standardised target (train mu/sigma)"
    reporting_space: str = "raw z-score units, after inverse_transform"

    # --- selection / reporting ------------------------------------------
    selection_metric: str = "validation missense Spearman"
    selection_rule: str = "maximise; undefined Spearman never counts as an improvement"
    tie_breakers: Tuple[str, ...] = ("raw-z MAE", "raw-z RMSE")
    composite_score: bool = False
    test_labels_used: bool = False

    # --- what is being compared -----------------------------------------
    objectives: Tuple[str, ...] = ("huber", "half_mse")
    huber_delta: float = HUBER_DELTA
    only_intentional_difference: str = "the loss function"

    status: str = (
        "SPECIFIED BUT NOT RUN -- the real token-level Stage 1 is an external "
        "dependency, so no Huber vs 0.5*MSE comparison exists and no loss has "
        "been selected."
    )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["objective_descriptions"] = dict(OBJECTIVE_DESCRIPTIONS)
        return d


#: The single instance to import. Do not construct variants of it.
PILOT = PilotProtocol()

#: Fields that MUST be identical between the two runs for the comparison to be
#: fair. A runner should assert this set matches before reporting anything.
FAIR_COMPARISON_FIELDS: Tuple[str, ...] = (
    "optimizer",
    "learning_rate",
    "weight_decay",
    "batch_size",
    "gradient_accumulation",
    "grad_clip_norm",
    "precision",
    "scheduler",
    "max_epochs",
    "early_stopping_patience",
    "seeds",
    "init_policy",
    "cohort",
    "split_source",
    "target_column",
    "scaler_fit_split",
    "scaler_std_ddof",
    "selection_metric",
    "selection_rule",
)


def assert_fair(a: PilotProtocol, b: PilotProtocol) -> None:
    """Refuse a comparison whose two arms differ in anything but the loss."""
    diff = {
        f: (getattr(a, f), getattr(b, f))
        for f in FAIR_COMPARISON_FIELDS
        if getattr(a, f) != getattr(b, f)
    }
    if diff:
        raise ValueError(
            "the two objective runs differ in more than the loss function: "
            f"{diff}. Task D forbids tuning one arm separately from the other."
        )
