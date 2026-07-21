#!/usr/bin/env bash
set -e

echo "RQ3 artifact verification"
echo "Expected environment: rq3"
echo

python - <<'PY'
import importlib

modules = ["torch", "torchvision", "numpy", "PIL", "skimage"]
for module in modules:
    importlib.import_module(module)
    print(f"ok import {module}")
PY

test -f code/full_sdar.py
test -f code/convnext_split.py
test -f code/data_tinyimagenet.py
test -f code/perturbations.py

test -f results/figures/rq3_qualitative_pair_3_9.pdf
test -f results/figures/probe_sheets/rq3_convnext_tiny_none_fixed_probe_orig_c1_c4.png
test -f results/figures/probe_sheets/rq3_convnext_tiny_q4_fixed_probe_orig_c1_c4.png
test -f results/figures/probe_sheets/rq3_convnext_tiny_q2_fixed_probe_orig_c1_c4.png
test -f results/figures/probe_sheets/rq3_convnext_tiny_noise0.01_fixed_probe_orig_c1_c4.png
test -f results/figures/probe_sheets/rq3_convnext_tiny_noise0.05_fixed_probe_orig_c1_c4.png

echo
echo "RQ3 artifact files verified."
echo "Full training requires Tiny ImageNet and is run from the source files in code/."
