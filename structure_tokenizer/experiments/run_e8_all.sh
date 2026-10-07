#!/bin/bash
# E8 전체(팔 b/c/d × seed 42–46 = 15회)를 4개씩 병렬로. 먼저 python e8_joint.py --pack 이 필요하다.
cd "$(dirname "$0")"
mkdir -p results/E8/_logs
for s in 42 43 44 45 46; do for a in b c d; do echo "$a $s"; done; done |
  xargs -P 4 -L 1 bash -c 'python e8_joint.py --arm $0 --seed $1 > results/E8/_logs/$0_seed$1.log 2>&1'
python e8_joint.py --analyze
