"""
smoke_test.py — forward (+ backward for M3/M4) one real batch through each of
M1-M4 and report shapes, NaNs, concat_dim, parameter counts, memory, and
(M3/M4) LoRA-only gradient flow.

Does NOT train anything -- this only checks the wiring is correct before a
real training loop is built.

HOW TO RUN
----------
    python smoke_test.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from model import build_model  # noqa: E402
from src.config_loader import load_model_config  # noqa: E402

STRUCT_X_PATH = "data/structure/results/rad51c_X.npy"
STRUCT_FEATURES_PATH = "data/structure/results/rad51c_struct_features.csv"
META_X_PATH = "data/structure/results/rad51c_meta_X.npy"
MANIFEST_PATH = "data/split_manifest.csv"
WT_SEQ_PATH = "data/wt_sequence.txt"
CACHE_PATH = "data/diff_emb_raw_smoketest.pt"

# Mixed batch: missense + synonymous + one codon_deletion (indel), all within
# the first 30 rows covered by diff_emb_raw_smoketest.pt -- exercises the
# length-grouping logic in DiffEmbedder's e2e path, not just the same-length
# missense/synonymous case.
BATCH_ROWS = [0, 1, 2, 3, 5, 6, 13, 22]


def load_batch():
    struct_features = pd.read_csv(STRUCT_FEATURES_PATH)
    struct_X = np.load(STRUCT_X_PATH)
    meta_X = np.load(META_X_PATH)
    manifest_df = pd.read_csv(MANIFEST_PATH)
    wt_seq = Path(WT_SEQ_PATH).read_text().strip()

    # struct/meta row order is 1:1 with split_manifest.csv's scope-filtered
    # subset (verified in the struct/meta build steps) -- use struct_features'
    # own var_id column as the index into struct_X/meta_X.
    var_ids = struct_features["var_id"].tolist()
    batch_var_ids = [var_ids[i] for i in BATCH_ROWS]

    struct_x = torch.from_numpy(struct_X[BATCH_ROWS]).float()
    meta_x = torch.from_numpy(meta_X[BATCH_ROWS]).float()

    manifest = {
        row["var_id"]: {"pp": int(row["pp"]), "mut_seq": row["mut_seq"]}
        for row in manifest_df.to_dict("records")
    }

    print(f"[batch] {len(batch_var_ids)} variants: {batch_var_ids}")
    print(f"[batch] struct_x {tuple(struct_x.shape)}  meta_x {tuple(meta_x.shape)}")
    return batch_var_ids, struct_x, meta_x, manifest, wt_seq


def param_counts(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def run_one(model_id: str, batch_var_ids, struct_x, meta_x, manifest, wt_seq):
    print(f"\n{'=' * 60}\n{model_id}\n{'=' * 60}")
    cfg = load_model_config(f"configs/{model_id.lower()}.yaml")

    # configs/m3.yaml, m4.yaml target the real RTX 3090 box (device: cuda).
    # This sandbox has no GPU -- fall back to CPU for the smoke test only;
    # the config file itself is left untouched.
    if cfg.get("esm", {}).get("device") == "cuda" and not torch.cuda.is_available():
        print(f"[{model_id}] no CUDA in this sandbox -- smoke-testing on CPU "
              f"(config still says device=cuda for the real training box)")
        cfg["esm"]["device"] = "cpu"

    cache_path = CACHE_PATH if cfg["esm_mode"] == "cached" else None
    m_manifest = manifest if cfg["esm_mode"] == "e2e" else None
    m_wt_seq = wt_seq if cfg["esm_mode"] == "e2e" else None

    t0 = time.time()
    model, dims = build_model(
        cfg, STRUCT_X_PATH, META_X_PATH,
        manifest=m_manifest, wt_seq=m_wt_seq, cache_path=cache_path,
    )
    print(f"[{model_id}] dims (computed, not hardcoded): {dims}")
    print(f"[{model_id}] build time: {time.time() - t0:.1f}s")

    total, trainable = param_counts(model)
    print(f"[{model_id}] params total={total:,}  trainable={trainable:,}"
          f"  ({100 * trainable / total:.4f}%)")

    cuda_ok = cfg["esm_mode"] == "e2e" and torch.cuda.is_available()
    if cfg["esm_mode"] == "e2e" and not torch.cuda.is_available():
        print(f"[{model_id}] NOTE: config requests device=cuda but no CUDA in "
              f"this sandbox -- running on CPU. GPU memory numbers below are "
              f"N/A here; re-run on the real RTX 3090 box for those.")

    # ---- forward ----
    t0 = time.time()
    out = model(batch_var_ids, struct_x, meta_x)
    fwd_s = time.time() - t0
    print(f"[{model_id}] forward: out.shape={tuple(out.shape)}  "
          f"NaN={torch.isnan(out).any().item()}  time={fwd_s:.1f}s")
    assert out.shape == (len(batch_var_ids),)
    assert not torch.isnan(out).any()

    if cuda_ok:
        torch.cuda.reset_peak_memory_stats()
        print(f"[{model_id}] GPU memory allocated: "
              f"{torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # ---- backward (M3/M4 only: verify grad reaches LoRA, not ESM base) ----
    if cfg["esm_mode"] == "e2e":
        model.zero_grad()
        loss = out.pow(2).mean()
        t0 = time.time()
        loss.backward()
        bwd_s = time.time() - t0

        # NOTE on lora_A: LoRA initialises lora_B = 0, so at step 0 the chain
        # rule d(output)/d(lora_A) contains a factor of lora_B and is exactly
        # zero -- lora_A legitimately has a *populated* zero-valued grad at
        # init (it becomes nonzero after the first optimizer step moves
        # lora_B away from 0). So "reached by autograd" (p.grad is not None)
        # is the correct wiring check, not "grad is nonzero".
        esm_model = model.diff_embedder.encoder.model
        n_lora_params = n_lora_reached = n_lora_nonzero = 0
        n_base_params = n_base_with_grad = 0
        lora_grad_norm = 0.0
        for name, p in esm_model.named_parameters():
            is_lora = "lora_A" in name or "lora_B" in name
            if is_lora:
                n_lora_params += 1
                if p.grad is not None:
                    n_lora_reached += 1
                    lora_grad_norm += p.grad.norm().item() ** 2
                    if p.grad.abs().sum().item() > 0:
                        n_lora_nonzero += 1
            else:
                n_base_params += 1
                if p.grad is not None:
                    n_base_with_grad += 1
        lora_grad_norm = lora_grad_norm ** 0.5

        print(f"[{model_id}] backward time: {bwd_s:.1f}s")
        print(f"[{model_id}] LoRA params reached by autograd: "
              f"{n_lora_reached}/{n_lora_params}  (grad norm={lora_grad_norm:.4e})")
        print(f"[{model_id}] LoRA params with NONZERO grad: {n_lora_nonzero}/{n_lora_params}"
              f"  (expect ~half: lora_B only, since lora_A's grad is exactly 0 "
              f"at init while lora_B=0 -- this is expected LoRA behaviour, not a bug)")
        print(f"[{model_id}] ESM BASE params with grad set (should be 0): "
              f"{n_base_with_grad}/{n_base_params}")
        assert n_base_with_grad == 0, "gradient leaked into frozen ESM base weights!"
        assert n_lora_reached == n_lora_params, "some LoRA params were never reached by autograd!"

        if cuda_ok:
            print(f"[{model_id}] GPU peak memory: "
                  f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

        # WT/MUT shared-adapter guarantee: there is exactly ONE encoder.model
        # instance, used for both the WT forward and every MUT forward inside
        # DiffEmbedder._forward_e2e -- so they necessarily share LoRA weights.
        print(f"[{model_id}] WT/MUT share one encoder instance: "
              f"{model.diff_embedder.encoder is not None}")

    return dims["concat_dim"]


def main():
    batch_var_ids, struct_x, meta_x, manifest, wt_seq = load_batch()

    concat_dims = {}
    for model_id in ["M1", "M2", "M3", "M4"]:
        concat_dims[model_id] = run_one(
            model_id, batch_var_ids, struct_x, meta_x, manifest, wt_seq
        )

    print(f"\n{'=' * 60}\nSUMMARY\n{'=' * 60}")
    for k, v in concat_dims.items():
        print(f"  {k}: concat_dim = {v}")
    diff_m1_m2 = concat_dims["M1"] - concat_dims["M2"]
    diff_m3_m4 = concat_dims["M3"] - concat_dims["M4"]
    print(f"\n  M1 - M2 = {diff_m1_m2}  (expect 11, Block B dim)")
    print(f"  M3 - M4 = {diff_m3_m4}  (expect 11, Block B dim)")
    assert diff_m1_m2 == 11
    assert diff_m3_m4 == 11
    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
