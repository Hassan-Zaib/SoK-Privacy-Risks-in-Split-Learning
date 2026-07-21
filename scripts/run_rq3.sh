#!/usr/bin/env bash
set -euo pipefail

source ~/miniconda/etc/profile.d/conda.sh
conda activate rq3

echo "RQ3 uses the rq3 environment."
echo "The reusable ConvNeXt-Tiny implementation is under src/rq3."
echo "Final processed metrics are under results/rq3/summaries."
echo "Final qualitative figures are under results/rq3/figures."
echo
echo "The original RQ3 training jobs were launched from the RQ3 experiment workspace."
echo "Use environment/rq3.yml and docs/dataset_setup.md to recreate the environment and dataset layout."
