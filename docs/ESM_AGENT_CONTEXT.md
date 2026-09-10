# ESM agent context

## Repo state and instruction file check

- Repository root: `/home/coder/deepCANCERsite`
- Current branch: `minseon/esm-module`
- Current HEAD: `02963a41e161c3ac6190809667387e130d63c732`
- Git status at the start of this work was clean aside from the workspace branch state; no destructive reset/rebase or file deletion was performed.
- Existing instruction file review: [CLAUDE.md](../CLAUDE.md) is present and active. No AGENTS file was present in the repository root.
- The guidance in [CLAUDE.md](../CLAUDE.md) and the project code agree on the repo-native ESM path and the 650M architecture; no contradictory branch-specific override was found.

## Verified environment

This was checked in the project virtual environment at `/home/coder/deepCANCERsite/.venv` without changing packages or re-installing PyTorch.

- Python: `3.10.12`
- PyTorch: `2.5.1+cu124`
- CUDA: available (`True`)
- CUDA version: `12.4`
- GPU: `Tesla V100-SXM2-32GB`
- GPU VRAM: `32,494 MB` (~32 GB)
- CPU count: `80` host-visible; cgroup CPU allocation reported as `max 100000` (this container is effectively sharing CPU but not locked to a fixed quota)
- Host-visible RAM: `515,830 MB total`, `452,417 MB available` from `free -m`
- Host-visible memory from `/proc/meminfo`: `MemTotal=528,210,824 kB`, `MemAvailable=459,744,532 kB`
- Container effective memory allocation: cgroup `memory.max=34,359,738,368` (32 GiB) and `memory.current=7,461,556,224` (approx. 7.0 GiB)
- Root disk: `5.2T total`, `2.8T available` on `/`
- Home volume: `4.94 GB total`, `0.59 GB free` (still below the safe threshold for checkpoint download)
- Repo-native ESM import path: `/home/coder/deepCANCERsite/esm/__init__.py`
- Verified absolute path from Python: `esm.__file__ == /home/coder/deepCANCERsite/esm/__init__.py`
- ESM model entrypoint present: `esm.pretrained.esm2_t33_650M_UR50D` exists

Resource note: the values from `free` and `/proc/meminfo` are host-visible values; the cgroup memory limit is the container effective allocation. We do not treat the host-visible numbers as a guaranteed per-workspace allocation limit.

## Safe operating rules for this workspace

1. Do not download the full `esm2_t33_650M_UR50D` checkpoint while the persistent home volume remains at ~5 GB; the check in `scripts/nrp_setup.sh` is intentionally not rerun here.
2. Do not upgrade PyTorch, downgrade Python, or reinstall CUDA. The current image already matches the working environment.
3. Prefer the repo-native `esm` package and local code paths; the minimal validation path is CPU-safe and checkpoint-free.
4. Keep all heavy ESM inference and checkpoint generation behind a later storage-expansion gate.

## Small-test path for Stage B work

The safe validation path is:

- use the repo-native `esm` package import as the source of truth
- run small, fixed-length sequence fixtures without invoking the 650M checkpoint download path
- confirm the identity and shape contracts in the embedding pipeline using local synthetic or small JSON/sequence fixtures
- only after storage is confirmed should a real 650M forward pass be run for cohort-wide extraction

## Research contract in this branch

- Use repo-native `esm` and not a pip-installed stand-alone `fair-esm` package.
- Maintain the ESM extraction contract `repr_layers=[31,32,33]` for the final working path.
- Keep the window selection contract consistent with the user research notes: WT 376 aa; missense/synonymous ±10; indels with ±10 flank windows.
- Do not mix gap handling with residue-level subtraction; handle residues and valid masks explicitly.
- Keep `H_WT/H_MUT/delta_H` shaped as `[B, 3, A, 1280]` and propagate `wt_pos/mut_pos`, validity masks, and `slot_kind` together.
- Keep Stage 1/2 separated from the existing pooling M1–M4 pipeline.

## Required next gate

The next checkpoint-enabled step is explicitly blocked until the storage issue is resolved. That is the reason no full ESM checkpoint download, cache build, or cohort-wide training has been run in this session.
