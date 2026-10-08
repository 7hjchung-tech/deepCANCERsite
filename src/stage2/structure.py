"""Structure-token interface and variant-ID-aligned structure table loading.

The learned StructureTokenizer (Q-PLE / per-feature projection / SS embedding /
field-ID embedding) is NOT in this repository yet. This module defines the
contract it must satisfy and refuses to run real training without one.

Tokenizer contract
------------------
  input  raw: {"continuous": FloatTensor [B, 8]  (order = CONTINUOUS_COLUMNS),
               "ss":         LongTensor  [B]     (0 helix, 1 sheet, 2 loop)}
  output S:   FloatTensor [B, 9, 32]   token j <-> STRUCT_TOKEN_NAMES[j]
  fit_preprocessing(train_raw): called once with TRAIN-split rows only; any
      data-estimated statistics (e.g. Q-PLE bin edges) are fitted there and
      then frozen. Validation/test rows are only ever passed to forward().
"""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .schema import CONTINUOUS_COLUMNS, N_STRUCT_TOKENS, SS_COLUMNS, STRUCT_TOKEN_DIM


class StructureTokenizerMissing(RuntimeError):
    """Raised when real training is requested without a tokenizer implementation."""


class StructureTokenizer(nn.Module):
    """Base class for the (not yet in repo) learned structure tokenizer."""

    out_tokens: int = N_STRUCT_TOKENS
    out_dim: int = STRUCT_TOKEN_DIM

    def fit_preprocessing(self, train_raw: dict) -> None:
        return None

    def forward(self, raw: dict) -> torch.Tensor:
        raise NotImplementedError


def load_tokenizer(spec: str | None, cfg: dict) -> StructureTokenizer:
    """`spec` is "package.module:factory". The factory receives cfg and must
    return a StructureTokenizer. Missing module/factory is a hard error."""
    if not spec:
        raise StructureTokenizerMissing(
            "No StructureTokenizer was provided. The learned tokenizer is not in this repository yet, "
            "so real Stage 2 training cannot start. Pass --tokenizer module:factory once it is available. "
            "(Synthetic tokenizers are only allowed in --synthetic-tokenizer smoke runs.)"
        )
    if ":" not in spec:
        raise ValueError(f"--tokenizer must look like module:factory, got {spec!r}")
    module_name, factory_name = spec.split(":", 1)
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        raise StructureTokenizerMissing(
            f"tokenizer module {module_name!r} was not found ({e}). Put the StructureTokenizer "
            f"implementation in the repo (or on PYTHONPATH) before real training."
        ) from e
    tok = getattr(module, factory_name)(cfg)
    if not isinstance(tok, StructureTokenizer):
        raise TypeError(f"{spec} must return a StructureTokenizer instance, got {type(tok).__name__}")
    return tok


class StructureStore:
    """Block A feature rows aligned to cohort variant IDs (never by row order)."""

    def __init__(self, var_ids: list[str], continuous: np.ndarray, ss: np.ndarray, splits: dict[str, str]):
        self.var_ids = list(var_ids)
        self._index = {v: i for i, v in enumerate(self.var_ids)}
        self.continuous = continuous.astype(np.float32)
        self.ss = ss.astype(np.int64)
        self.splits = splits

    def raw(self, ids: list[str]) -> dict:
        idx = np.array([self._index[v] for v in ids], dtype=np.int64)
        return {
            "continuous": torch.as_tensor(self.continuous[idx]),
            "ss": torch.as_tensor(self.ss[idx]),
        }

    def ids_in_split(self, split: str) -> list[str]:
        return [v for v in self.var_ids if self.splits[v] == split]


def load_structure_store(table_path: str | Path, cohort_var_ids: list[str],
                         manifest_split: dict[str, str]) -> StructureStore:
    df = pd.read_csv(table_path)
    required = ["var_id", "split", *CONTINUOUS_COLUMNS, *SS_COLUMNS]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"{table_path}: missing columns {missing_cols}")

    dup = df["var_id"][df["var_id"].duplicated()].unique().tolist()
    if dup:
        raise ValueError(f"{table_path}: duplicated var_id values (first 5): {dup[:5]}")
    df = df.set_index("var_id", drop=False)

    not_found = [v for v in cohort_var_ids if v not in df.index]
    if not_found:
        raise ValueError(f"{table_path}: {len(not_found)} cohort variants have no structure row "
                         f"(first 5: {not_found[:5]})")
    extra = len(set(df.index) - set(cohort_var_ids))

    sub = df.loc[cohort_var_ids]
    if sub[list(CONTINUOUS_COLUMNS)].isna().any().any():
        bad = sub.index[sub[list(CONTINUOUS_COLUMNS)].isna().any(axis=1)].tolist()
        raise ValueError(f"{table_path}: NaN in continuous Block A features for {len(bad)} variants")
    ss_onehot = sub[list(SS_COLUMNS)].to_numpy()
    if not np.all(np.isin(ss_onehot, [0, 1])) or not np.all(ss_onehot.sum(axis=1) == 1):
        raise ValueError(f"{table_path}: secondary-structure one-hot columns are not exactly one-hot")
    ss_code = ss_onehot.argmax(axis=1)

    mismatch = [v for v in cohort_var_ids if manifest_split.get(v) != sub.loc[v, "split"]]
    if mismatch:
        raise ValueError(f"{table_path}: split disagrees with split_manifest for {len(mismatch)} variants")

    if extra:
        print(f"[stage2.structure] {extra} structure rows are not in the cohort and were ignored")
    splits = {v: sub.loc[v, "split"] for v in cohort_var_ids}
    return StructureStore(
        cohort_var_ids,
        sub[list(CONTINUOUS_COLUMNS)].to_numpy(dtype=np.float32),
        ss_code,
        splits,
    )


class QkTokenizerAdapter(StructureTokenizer):
    """Wraps structure_tokenizer/tokenizer.py (Qk: PLE + FT-Transformer tokens) without modifying it.

    The inner module's bin edges are estimated from data, so it is created inside
    fit_preprocessing(train_raw) using TRAIN rows only; forward() before that is an error.
    """

    def __init__(self, d_s: int = STRUCT_TOKEN_DIM, n_bins: int = 4) -> None:
        super().__init__()
        self.d_s = d_s
        self.n_bins = n_bins
        self.inner = None

    def fit_preprocessing(self, train_raw: dict) -> None:
        from structure_tokenizer.tokenizer import StructureTokenizer as QkModule, fit_qk_bins

        cont = train_raw["continuous"].detach().cpu().numpy()
        self.inner = QkModule(fit_qk_bins(cont, self.n_bins), self.d_s)

    def forward(self, raw: dict) -> torch.Tensor:
        if self.inner is None:
            raise RuntimeError("QkTokenizerAdapter.fit_preprocessing(train_raw) must run before forward")
        return self.inner(raw["continuous"], raw["ss"])


def make_qk_tokenizer(cfg: dict) -> QkTokenizerAdapter:
    return QkTokenizerAdapter(d_s=int(cfg.get("d_s", STRUCT_TOKEN_DIM)), n_bins=int(cfg.get("n_bins", 4)))
