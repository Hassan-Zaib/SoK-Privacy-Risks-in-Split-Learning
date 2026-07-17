#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

for cut in c1 c2 c3 c4; do
  echo "===== SDAR Tiny ImageNet K=10 Dirichlet a=0.5 PSL SGLR separate-front ${cut} ====="
  python scripts/run_sdar_psl_tiny_dirichlet_a05_sglr.py \
    --dataset tinyimagenet \
    --model resnet18 \
    --cut "${cut}" \
    --num-clients 10 \
    --client-batch-size 32 \
    --num-iter 20000 \
    --eval-batches 4 \
    --alpha 0.5 \
    --seed 42 \
    --splitlr-alpha 0.5 \
    --splitavg-phi 0.5
done
