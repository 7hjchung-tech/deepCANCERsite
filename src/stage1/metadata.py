"""Common edit metadata token: featurization + train-only scaler.

Only information derivable from the sequence/edit itself is used (per the
task spec): normalized edit type, anchor (= preserved prefix length u),
n_del, n_ins, WT/MUT length. z-score, functional class, split id and var_id
are never fed in here.

`translation_status` is recorded on NormalizedEdit for bookkeeping but is
constant across the currently supported cohort, so it is deliberately NOT
one of the trainable feature dimensions below (a constant input carries no
information for a learned model).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .alignment import NormalizedEdit
from .schema import EDIT_TYPES, RAW_META_DIM

# raw feature layout: [one-hot edit_type (5)] + [u, d, m, wt_len, mut_len]
_NUMERIC_START = len(EDIT_TYPES)


def raw_meta_features(edit: NormalizedEdit) -> np.ndarray:
    onehot = [1.0 if edit.edit_type == t else 0.0 for t in EDIT_TYPES]
    numeric = [float(edit.u), float(edit.d), float(edit.m), float(edit.wt_len), float(edit.mut_len)]
    vec = np.asarray(onehot + numeric, dtype=np.float32)
    assert vec.shape == (RAW_META_DIM,)
    return vec


@dataclass
class MetaScaler:
    """Standardizes the numeric half of raw_meta_features using TRAIN-ONLY stats.

    The one-hot edit-type half is left untouched (mirrors train.py's
    standardize_struct treatment of binary/one-hot columns).
    """

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, raw_matrix: np.ndarray) -> "MetaScaler":
        numeric = raw_matrix[:, _NUMERIC_START:]
        mean = numeric.mean(axis=0)
        std = numeric.std(axis=0)
        std[std == 0] = 1.0
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32))

    def transform(self, raw_matrix: np.ndarray) -> np.ndarray:
        out = raw_matrix.copy()
        out[:, _NUMERIC_START:] = (out[:, _NUMERIC_START:] - self.mean) / self.std
        return out

    def state_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_state_dict(cls, state: dict) -> "MetaScaler":
        return cls(mean=np.asarray(state["mean"], dtype=np.float32), std=np.asarray(state["std"], dtype=np.float32))
