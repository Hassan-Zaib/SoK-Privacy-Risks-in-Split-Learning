#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python run_maera.py \
  --cut c1 \
  --batch-size 128 \
  --main-epochs 1 \
  --shadow-epochs 1 \
  --inverse-epochs 1 \
  --max-smashed-batches 2 \
  --n-attack 8 \
  --no-lpips
