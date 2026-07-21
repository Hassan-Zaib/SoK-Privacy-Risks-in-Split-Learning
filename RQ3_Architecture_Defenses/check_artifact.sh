#!/usr/bin/env bash
set -euo pipefail

echo "RQ3 artifact verification"
echo "Expected environment: rq3"
echo

python - <<'PY'
import importlib
for name in ["torch", "torchvision", "numpy", "PIL", "skimage"]:
    importlib.import_module(name)
    print(f"ok import {name}")
PY

test -f code/full_sdar.py
test -f code/train_utils.py
test -f scripts/train_split_convnext.py
test -f scripts/run_sdar_on_convnext_smashed.py

test -f results/figures/rq3_qualitative_pair_3_9.pdf
test -f results/figures/probe_sheets/rq3_convnext_tiny_none_fixed_probe_orig_c1_c4.png
test -f results/figures/probe_sheets/rq3_convnext_tiny_q4_fixed_probe_orig_c1_c4.png
test -f results/figures/probe_sheets/rq3_convnext_tiny_q2_fixed_probe_orig_c1_c4.png
test -f results/figures/probe_sheets/rq3_convnext_tiny_noise0.01_fixed_probe_orig_c1_c4.png
test -f results/figures/probe_sheets/rq3_convnext_tiny_noise0.05_fixed_probe_orig_c1_c4.png

echo
echo "RQ3 artifact files verified."
