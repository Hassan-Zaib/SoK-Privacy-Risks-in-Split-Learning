#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
for cut in c1 c2 c3 c4; do
  python model_inversion_resnet18.py --cut "$cut" "$@"
done
