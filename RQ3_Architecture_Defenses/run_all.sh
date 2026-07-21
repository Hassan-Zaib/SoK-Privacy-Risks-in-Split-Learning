#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

echo "RQ3 full reproduction"
echo "Expected environment: rq3"
echo "Requires Tiny ImageNet at data/tiny-imagenet-200 or set DATA_ROOT=/path/to/tiny-imagenet-200"
echo

bash scripts/run_rq3_sequential.sh
