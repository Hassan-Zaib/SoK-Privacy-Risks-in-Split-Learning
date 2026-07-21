#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

for cut in c1 c2 c3 c4; do
  echo "===== SDAR Tiny ImageNet K=10 IID round-robin ${cut} ====="
  python scripts/run_sdar_rr_tiny_iid.py \
    --dataset tinyimagenet \
    --model resnet18 \
    --cut "${cut}" \
    --num-clients 10 \
    --batch-size 128 \
    --num-iter 20000 \
    --eval-batches 1
done
