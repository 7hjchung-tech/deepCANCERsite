#!/usr/bin/env bash
# nrp_setup.sh — one-shot environment setup inside an NRP/Nautilus Coder workspace
# (Templates > Cuda/Pytorch/TensorFlow).
#
#   bash scripts/nrp_setup.sh
#
# What it does
#   1. reports GPU / disk / preinstalled torch
#   2. creates .venv with --system-site-packages so the image's CUDA torch is
#      reused instead of downloading a ~2.5 GB wheel into a 5 GB volume
#   3. installs only the packages the image is missing
#   4. downloads the ESM-2 650M checkpoint (~2.5 GB) into the persistent home
#   5. verifies torch.cuda + an ESM forward pass end to end
#
# Safe to re-run: every step is idempotent.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
VENV="$ROOT/.venv"

echo "=== [0/5] workspace ==============================================="
echo "repo      : $ROOT"
echo "python    : $(python3 --version 2>&1)"
df -h "$HOME" | tail -1 | awk '{print "home disk : "$2" total, "$4" free ("$5" used)"}'
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
        | sed 's/^/gpu       : /'
else
    echo "gpu       : nvidia-smi NOT FOUND — this workspace has no GPU attached."
    echo "            Stop the workspace, edit its parameters (GPUs >= 1), restart."
fi

echo
echo "=== [1/5] venv (--system-site-packages) ==========================="
if [ ! -d "$VENV" ]; then
    python3 -m venv --system-site-packages "$VENV"
    echo "created $VENV"
else
    echo "$VENV already exists — reusing"
fi
PY="$VENV/bin/python"
"$PY" -m pip install --quiet --upgrade pip

echo
echo "=== [2/5] torch ==================================================="
if "$PY" -c "import torch" 2>/dev/null; then
    echo "torch     : $("$PY" -c 'import torch; print(torch.__version__)') (from image, not reinstalling)"
else
    echo "torch not in image — installing CUDA 12.1 build (~2.5 GB download)"
    "$PY" -m pip install torch --index-url https://download.pytorch.org/whl/cu121
fi

echo
echo "=== [3/5] remaining runtime deps =================================="
# Only what dump_diff_emb.py / smoke_test.py / model.py actually import.
# scipy, scikit-learn, matplotlib, biotite, transformers are for the analysis
# and structure-feature pipelines — add them with requirements.txt if needed.
for pkg in numpy pandas yaml; do
    if ! "$PY" -c "import $pkg" 2>/dev/null; then MISSING="${MISSING:-} $pkg"; fi
done
if [ -n "${MISSING:-}" ]; then
    echo "installing:${MISSING/yaml/pyyaml}"
    # shellcheck disable=SC2086
    "$PY" -m pip install ${MISSING/yaml/pyyaml}
else
    echo "numpy / pandas / pyyaml already present"
fi

echo
echo "=== [4/5] ESM-2 650M checkpoint ==================================="
CKPT="$HOME/.cache/torch/hub/checkpoints/esm2_t33_650M_UR50D.pt"
if [ -f "$CKPT" ]; then
    echo "already downloaded: $CKPT ($(du -h "$CKPT" | cut -f1))"
else
    echo "downloading ~2.5 GB to $CKPT ..."
    "$PY" - <<'PYCODE'
import esm
esm.pretrained.esm2_t33_650M_UR50D()
print("checkpoint downloaded")
PYCODE
fi

echo
echo "=== [5/5] verification ============================================"
"$PY" - <<'PYCODE'
import torch
print(f"torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device: {torch.cuda.get_device_name(0)} "
          f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB)")
else:
    print("WARNING: no CUDA — dump_diff_emb.py will fall back to CPU (hours, not minutes)")

from src.embeddings.esm_encoder import ESMEncoder
dev = "cuda" if torch.cuda.is_available() else "cpu"
enc = ESMEncoder({"device": dev, "repr_layer": 33})
h = enc.encode(["MKTAYIAKQR"], repr_layers=33)[33]
print(f"ESM forward OK on {dev}: {tuple(h.shape)}  (expect (1, 10, 1280))")
PYCODE

echo
echo "Setup complete. Activate with:  source .venv/bin/activate"
echo "Next:  python dump_diff_emb.py --device cuda --batch-size 32"
