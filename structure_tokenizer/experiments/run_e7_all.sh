#!/bin/bash
# E7 전체 작업 10개를 4개씩 병렬로. 주 분석(44 tok)을 맨 앞에.
cd "$(dirname "$0")"
mkdir -p results/E7/stage2/_logs
printf "%s\n" "44 tok" "44 z" "42 tok" "43 tok" "45 tok" "46 tok" "42 z" "43 z" "45 z" "46 z" |
  xargs -P 4 -L 1 bash -c 'python e7_stage2.py --s1seed $0 --query $1 > results/E7/stage2/_logs/s$0_$1.log 2>&1'
echo ALL_DONE
