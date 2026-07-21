#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
for cut in c1 c2 c3 c4; do
  python train_resnet18_split.py --cut "$cut" "$@"
done
