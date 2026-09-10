#!/usr/bin/env python3
"""Project-local ESM environment probe.

This script intentionally avoids downloading the 650M checkpoint or training
anything. It records the exact interpreter, GPU/CUDA, CPU/RAM, disk, and the
repo-native esm import path so the research setup remains reproducible without
changing the current environment.
"""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def run_cmd(args: list[str]) -> str:
    try:
        out = subprocess.check_output(args, cwd=str(REPO_ROOT), text=True, stderr=subprocess.STDOUT)
        return out.strip()
    except Exception:
        return ""


def main() -> int:
    git_branch = run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    git_head = run_cmd(["git", "rev-parse", "HEAD"])
    git_status = run_cmd(["git", "status", "--short", "--branch"])

    esm_spec = importlib.util.find_spec("esm")
    esm_origin = esm_spec.origin if esm_spec is not None else None
    esm_file = None
    esm_pretrained_file = None
    esm_has_650m = False
    esm_has_mod = False
    if esm_spec is not None:
        try:
            import esm  # type: ignore
            esm_has_mod = True
            esm_file = getattr(esm, "__file__", None)
            esm_pretrained_file = getattr(esm.pretrained, "__file__", None)
            esm_has_650m = hasattr(esm.pretrained, "esm2_t33_650M_UR50D")
        except Exception:
            esm_has_mod = False
            esm_has_650m = False

    gpu_name = ""
    gpu_vram_mb = 0
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        gpu_name = gpu.name
        gpu_vram_mb = int(gpu.total_memory / (1024 ** 2))

    disk_root = shutil.disk_usage("/")
    disk_home = shutil.disk_usage(str(Path.home())) if Path.home().exists() else None

    payload = {
        "repo_root": str(REPO_ROOT),
        "git": {
            "branch": git_branch,
            "head": git_head,
            "status": git_status,
        },
        "python": {
            "version": sys.version.split()[0],
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "torch": {
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "gpu_name": gpu_name,
            "gpu_vram_mb": gpu_vram_mb,
            "gpu_driver": run_cmd(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
        },
        "cpu": {
            "count": os.cpu_count(),
            "model": platform.processor(),
        },
        "memory": {
            "total_mb": int(subprocess.check_output(["free", "-m"]).decode().splitlines()[1].split()[1]),
            "available_mb": int(subprocess.check_output(["free", "-m"]).decode().splitlines()[1].split()[6]),
        },
        "disk": {
            "root_total_gb": round(disk_root.total / (1024 ** 3), 2),
            "root_free_gb": round(disk_root.free / (1024 ** 3), 2),
            "home_total_gb": round(disk_home.total / (1024 ** 3), 2) if disk_home else None,
            "home_free_gb": round(disk_home.free / (1024 ** 3), 2) if disk_home else None,
        },
        "esm": {
            "spec_found": esm_spec is not None,
            "origin": esm_origin,
            "__file__": esm_file,
            "pretrained_file": esm_pretrained_file,
            "importable": esm_has_mod,
            "has_esm2_t33_650M_UR50D": esm_has_650m,
        },
        "guardrails": {
            "checkpoint_download_allowed": False,
            "reason": "The workspace still has a 5 GB home volume; the 650M checkpoint is ~2.5 GB and should not be downloaded until storage expansion is confirmed.",
            "upgrade_allowed": False,
            "reason_upgrade": "CUDA/PyTorch already match the image; no reinstall or version changes are required for the safe probe/test path.",
        },
    }

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
