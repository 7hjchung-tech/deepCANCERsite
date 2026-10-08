"""Shared setup and per-sample measurements for the Stage 2 diagnosis (diagnose_stage2.py,
instrument_stage2.py, train_stage2_reverse.py).

Nothing here changes training code paths. The data/Stage 1/tokenizer setup mirrors
train_stage2.py exactly (same cohort join, same train-only tokenizer fit, same seeding order)
so that reconstructed models are identical to the ones the recorded runs started from.
Test rows are never loaded into the diagnostic sample.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.stage1.cache import RawStage1Cache
from src.stage1.checkpoint import default_reference
from src.stage1.dataset import build_cohort, join_cohort_with_cache
from src.stage1.engine import group_of, set_all_seeds
from src.stage1.positional import build_position_encoding

from .engine import make_loader
from .model import Stage2Model
from .stage1_adapter import Stage1Handle, load_stage1_handle
from .structure import StructureStore, load_structure_store, make_qk_tokenizer

ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = ROOT / "configs" / "stage2" / "base.yaml"
DEFAULT_STAGE1 = "runs/stage1_v2/unified_reference_delta/W10/shipped_split/seed44/best.pt"


def sha16(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


@dataclass
class Setup:
    cfg: dict
    cache: RawStage1Cache
    by_split: dict[str, list[dict]]
    store: StructureStore
    handle: Stage1Handle
    window: int
    layers: list[int]
    device: str


def load_setup(stage1_ckpt: str = DEFAULT_STAGE1, device: str = "cuda", splits=("train", "val"),
               manifest: str = "data/split_manifest.csv", wt_seq: str = "data/wt_sequence.txt",
               cache_path: str = "data/stage1/raw_cache.pt",
               structure_table: str = "data/structure/results/rad51c_struct_features.csv") -> Setup:
    cfg = yaml.safe_load(BASE_CONFIG.read_text())
    cache = RawStage1Cache.load(str(ROOT / cache_path))
    man = pd.read_csv(ROOT / manifest)
    seq = (ROOT / wt_seq).read_text().strip()
    cohort = join_cohort_with_cache(build_cohort(man.to_dict("records"), seq), cache)
    by_split = {"train": [], "val": [], "test": []}
    for e in cohort.supported:
        by_split[e["row"]["split"]].append(e)
    manifest_split = {e["var_id"]: e["row"]["split"] for e in cohort.supported}
    store = load_structure_store(ROOT / structure_table, [e["var_id"] for e in cohort.supported], manifest_split)
    ref = default_reference(cfg.get("esm_checkpoint", "esm2_t33_650M_UR50D"), cache.wt_hash)
    handle = load_stage1_handle(str(ROOT / stage1_ckpt), ref, device)
    # test entries are dropped here so no diagnostic can touch them by accident
    by_split = {k: v for k, v in by_split.items() if k in splits}
    return Setup(cfg, cache, by_split, store, handle, int(handle.cfg["window_radius"]),
                 list(handle.cfg["layers"]), device)


def build_initial(setup: Setup, query_mode: str, seed: int, tau_init) -> tuple[Stage2Model, torch.nn.Module]:
    """Same call order as train_stage2.py: set_all_seeds -> tokenizer.fit -> Stage2Model."""
    set_all_seeds(seed)
    tok = make_qk_tokenizer(setup.cfg)
    tok.fit_preprocessing(setup.store.raw([e["var_id"] for e in setup.by_split["train"]]))
    m = Stage2Model(query_mode, tau_init=tau_init)
    return m.to(setup.device), tok.to(setup.device)


def load_trained(setup: Setup, ckpt_path: str | Path) -> tuple[Stage2Model, torch.nn.Module, dict]:
    ck = torch.load(ckpt_path, map_location=setup.device, weights_only=False)
    tok = make_qk_tokenizer(setup.cfg)
    tok.fit_preprocessing(setup.store.raw([e["var_id"] for e in setup.by_split["train"]]))
    tok.load_state_dict(ck["tokenizer"])       # bins are buffers: a mismatch would raise here
    m = Stage2Model(ck["query_mode"], tau_init=ck["cfg"].get("tau_init"))
    m.load_state_dict(ck["stage2"])
    return m.to(setup.device).eval(), tok.to(setup.device).eval(), ck


def load_trained_any(setup: Setup, ckpt_path: str | Path) -> tuple[torch.nn.Module, torch.nn.Module, dict]:
    """Like load_trained, but also accepts train_stage2_reverse.py checkpoints (key "mode"
    instead of "query_mode", R0/R1/R2 via src.stage2.reverse.build_reverse_model)."""
    ck = torch.load(ckpt_path, map_location=setup.device, weights_only=False)
    tok = make_qk_tokenizer(setup.cfg)
    tok.fit_preprocessing(setup.store.raw([e["var_id"] for e in setup.by_split["train"]]))
    tok.load_state_dict(ck["tokenizer"])
    if "mode" in ck:
        from .reverse import build_reverse_model
        m = build_reverse_model(ck["mode"], tau_init=ck["cfg"].get("tau_init"))
    else:
        m = Stage2Model(ck["query_mode"], tau_init=ck["cfg"].get("tau_init"))
    m.load_state_dict(ck["stage2"])
    return m.to(setup.device).eval(), tok.to(setup.device).eval(), ck


def diag_sample(entries: list[dict], n_per_type: int, seed: int = 0) -> list[dict]:
    """Fixed, type-stratified sample. Within a type, positions are spread by sorting on
    anchor position and taking evenly spaced rows after a seeded shuffle tie-break."""
    rng = np.random.default_rng(seed)
    out = []
    for g in ("missense", "synonymous", "indel"):
        es = [e for e in entries if group_of(e["edit"].edit_type) == g]
        order = rng.permutation(len(es))
        es = sorted([es[i] for i in order], key=lambda e: e["edit"].u)
        if len(es) <= n_per_type:
            out += es
        else:
            idx = np.linspace(0, len(es) - 1, n_per_type).round().astype(int)
            out += [es[i] for i in idx]
    return out


def loader_for(setup: Setup, entries: list[dict], batch_size: int = 64, shuffle: bool = False,
               y1_override: dict | None = None, store_override: StructureStore | None = None,
               handle_override: Stage1Handle | None = None):
    """store_override: use a different StructureStore than setup.store (e.g. a FixedStructureStore
    for a structure-ablation arm). handle_override: use a different (e.g. freshly reloaded)
    Stage1Handle than setup.handle -- needed whenever a caller mutates Stage1's weights (joint
    training) and must not let that leak into another run sharing the same Setup. meta_scaler
    comes from this handle, exactly as with setup.handle. Both default to None (existing
    behaviour, setup.store / setup.handle)."""
    h = handle_override if handle_override is not None else setup.handle
    return make_loader(entries, setup.cache, setup.window, setup.layers, h,
                       store_override if store_override is not None else setup.store,
                       "unified_reference_delta", batch_size=batch_size, shuffle=shuffle, y1_override=y1_override)


@torch.no_grad()
def stage1_components(handle: Stage1Handle, batch: dict) -> dict:
    """Re-derives the Stage 1 key decomposition K_res = content + pe + layer_emb and checks
    that it reproduces the K returned by the model (assert below)."""
    m = handle.model
    dev = next(m.parameters()).device
    b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
    out = m(b, return_extras=True)
    H_wt = b["H_wt"]
    B, Lyr, A, _ = H_wt.shape
    content = m.content_builder(H_wt, b["H_mut"], b["delta"], b["wt_present"], b["mut_present"],
                                b["delta_valid"], b["slot_kind"])
    pe = build_position_encoding(b["anchor_rel_coord"], b["insertion_rank"], (b["slot_kind"] == 2).float())
    pe = pe.unsqueeze(1).expand(B, Lyr, A, -1)
    lid = torch.as_tensor(b["layers"], device=dev, dtype=torch.long)
    lemb = m.layer_embedding(lid).view(1, Lyr, 1, -1).expand(B, Lyr, A, -1)
    K_res = (content + pe + lemb).reshape(B, Lyr * A, -1)
    assert torch.allclose(K_res, out["K"][:, :Lyr * A], atol=1e-5), "K decomposition does not match Stage 1"
    return {
        "K": out["K"], "V": out["V"], "valid": out["attention_valid"], "s1_weights": out["attn_weights"],
        "content": content.reshape(B, Lyr * A, -1), "pe": pe.reshape(B, Lyr * A, -1),
        "layer_emb": lemb.reshape(B, Lyr * A, -1), "n_res": Lyr * A,
        "slot_kind": b["slot_kind"], "y1": out["pred"] * handle.y_std + handle.y_mean,
    }


def entropy_rows(w: np.ndarray, n_valid: int) -> tuple[float, float, float, float]:
    """w: one distribution over valid tokens. Returns H, H/log(N) (nan if N==1), max p, KL(w||uniform)."""
    w = w[w > 0]
    H = float(-(w * np.log(w)).sum())
    norm = H / math.log(n_valid) if n_valid > 1 else float("nan")
    kl = math.log(n_valid) - H
    return H, norm, float(w.max()), kl


def offdiag_cos(X: torch.Tensor) -> torch.Tensor:
    """X [n, d] -> off-diagonal cosine values (flattened)."""
    Xn = torch.nn.functional.normalize(X, dim=-1, eps=1e-8)
    C = Xn @ Xn.T
    n = C.shape[0]
    return C[~torch.eye(n, dtype=torch.bool, device=C.device)]


def centered_var_ratio(X: torch.Tensor) -> float:
    """Fraction of token energy that varies across tokens: mean||X_i - mean X||^2 / mean||X_i||^2."""
    num = (X - X.mean(0, keepdim=True)).pow(2).sum(-1).mean()
    den = X.pow(2).sum(-1).mean().clamp_min(1e-12)
    return float(num / den)
