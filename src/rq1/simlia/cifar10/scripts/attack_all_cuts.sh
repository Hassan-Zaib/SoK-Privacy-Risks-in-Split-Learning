#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

for cut in c1 c2 c3 c4; do
  python run_simlia.py --cut "$cut" "$@"
done
