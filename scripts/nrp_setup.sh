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
echo "=== [1/5] python environment ======================================"
# Preferred: a venv with --system-site-packages, so the image's CUDA torch is
# reused instead of downloading a multi-GB wheel. Some NRP images ship without
# python3-venv (ensurepip missing), so fall back rather than dying: apt-get it
# if sudo works, otherwise use the system interpreter with `pip install --user`
# (packages land in ~/.local, which is on the persistent volume either way).
PY=""
PIP_USER=""

# A venv is only usable if it also has a working pip. A failed `python3 -m venv`
# still leaves bin/python behind, so testing for the interpreter alone would
# happily "reuse" a broken directory on the next run.
venv_ok() {
    [ -x "$VENV/bin/python" ] && "$VENV/bin/python" -m pip --version >/dev/null 2>&1
}

if venv_ok; then
    PY="$VENV/bin/python"
    echo "reusing existing venv: $VENV"
else
    if [ -e "$VENV" ]; then
        echo "discarding unusable venv at $VENV (no working pip)"
        rm -rf "$VENV"
    fi
    if python3 -m venv --system-site-packages "$VENV" 2>/dev/null && venv_ok; then
        PY="$VENV/bin/python"
        echo "created venv: $VENV"
    else
        rm -rf "$VENV"
        echo "python3 -m venv is unavailable in this image (ensurepip missing)."
        if sudo -n true 2>/dev/null; then
            echo "trying: sudo apt-get install -y python3-venv"
            if sudo apt-get update -qq \
                && sudo apt-get install -y -qq "python3.$(python3 -c 'import sys; print(sys.version_info[1])')-venv" \
                && python3 -m venv --system-site-packages "$VENV" 2>/dev/null \
                && venv_ok; then
                PY="$VENV/bin/python"
                echo "created venv: $VENV"
            else
                rm -rf "$VENV"
                echo "apt-get route did not work either"
            fi
        else
            echo "passwordless sudo not available — skipping the apt-get route"
        fi
    fi
fi

if [ -z "$PY" ]; then
    if ! python3 -m pip --version >/dev/null 2>&1; then
        echo "ERROR: neither venv nor pip is usable in this image."
        echo "       Ask in the Nautilus Support channel, or pick a different"
        echo "       Coder template (Cuda/Pytorch rather than TensorFlow)."
        exit 1
    fi
    PY="python3"
    PIP_USER="--user"
    echo "no venv — using the system interpreter with 'pip install --user'"
    echo "(installs go to ~/.local, which persists across workspace restarts)"
fi
# shellcheck disable=SC2086
"$PY" -m pip install --quiet --upgrade pip $PIP_USER 2>/dev/null || true

echo
echo "=== [2/5] torch ==================================================="
if "$PY" -c "import torch" 2>/dev/null; then
    echo "torch     : $("$PY" -c 'import torch; print(torch.__version__)') (from image, not reinstalling)"
else
    echo "torch not in image — installing CUDA 12.1 build (~2.5 GB download)"
    # shellcheck disable=SC2086
    "$PY" -m pip install $PIP_USER torch --index-url https://download.pytorch.org/whl/cu121
fi

echo
echo "=== [3/5] remaining runtime deps =================================="
# What dump_diff_emb.py / build_meta_x.py / train.py / tests/ actually import.
# matplotlib, biotite and transformers are only needed by the structure-feature
# and baseline pipelines — install those with requirements.txt if you need them.
declare -a WANT=(numpy pandas yaml sklearn pytest)
declare -a PIP_NAME=(numpy pandas pyyaml scikit-learn pytest)
MISSING=()
for i in "${!WANT[@]}"; do
    "$PY" -c "import ${WANT[$i]}" 2>/dev/null || MISSING+=("${PIP_NAME[$i]}")
done
if [ ${#MISSING[@]} -gt 0 ]; then
    echo "installing: ${MISSING[*]}"
    # shellcheck disable=SC2086
    "$PY" -m pip install $PIP_USER "${MISSING[@]}"
else
    echo "numpy / pandas / pyyaml / scikit-learn / pytest already present"
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
if [ "$PY" = "python3" ]; then
    echo "Setup complete. No venv to activate — just use 'python3'."
else
    echo "Setup complete. Activate with:  source .venv/bin/activate"
fi
echo "Next:"
echo "  python build_meta_x.py"
echo "  python dump_diff_emb.py --device cuda --batch-size 32"
echo "  python train.py --config configs/m1.yaml"
