#!/usr/bin/env bash
# E1 (천장·바닥) + E3 (인코딩 5가지 비교) 를 처음부터 끝까지. 먼저 ../build_dataset.py 로 데이터를 만든다.
#   bash run_e1_e3.sh                     # CPU 코어 수만큼 병렬
#   WORKERS=12 DEVICE=cuda bash run_e1_e3.sh
# 중간에 끊겨도 다시 실행하면 끝난 작업은 건너뛴다 (탐색 기록은 작업별 SQLite).
set -euo pipefail
cd "$(dirname "$0")"
WORKERS=${WORKERS:-$(python -c "import os; print(len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count())")}
DEVICE=${DEVICE:-cpu}
mkdir -p results
python -m pytest tests -q
# 탐색 횟수는 보고서와 같게: GBDT 100회, 인코딩 후보 50회
python launch.py --groups E1 --workers "$WORKERS" --device "$DEVICE" --n_trials 100 > results/launch_E1.log 2>&1
python launch.py --groups E3 --workers "$WORKERS" --device "$DEVICE" --n_trials 50 > results/launch_E3.log 2>&1
python compare.py
