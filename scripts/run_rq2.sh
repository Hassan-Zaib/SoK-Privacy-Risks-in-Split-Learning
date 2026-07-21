#!/usr/bin/env bash
set -euo pipefail

source ~/miniconda/etc/profile.d/conda.sh
conda activate splitlearn

echo "RQ2 uses the splitlearn environment."
echo "Run the following commands from the repository root."

echo
echo "Round-robin K10 IID"
echo "cd src/rq2/sdar_tiny_k10_RRiid && bash scripts/run_tiny_k10_rr_iid_all_cuts.sh"

echo
echo "Round-robin K10 NonIID"
echo "cd src/rq2/sdar_tiny_k10_RRnoniid && bash scripts/run_tiny_k10_rr_noniid_all_cuts.sh"

echo
echo "Round-robin K10 Dirichlet"
echo "cd src/rq2/sdar_tiny_k10_RRdirichlet_a05 && bash scripts/run_tiny_k10_rr_dirichlet_a05_all_cuts.sh"

echo
echo "Round-robin K20 IID"
echo "cd src/rq2/sdar_tiny_k20_RRiid && bash scripts/run_tiny_k20_rr_iid_all_cuts.sh"

echo
echo "Round-robin K20 NonIID"
echo "cd src/rq2/sdar_tiny_k20_RRnoniid && bash scripts/run_tiny_k20_rr_noniid_all_cuts.sh"

echo
echo "Round-robin K20 Dirichlet"
echo "cd src/rq2/sdar_tiny_k20_RRdirichlet_a05 && bash scripts/run_tiny_k20_rr_dirichlet_a05_all_cuts.sh"

echo
echo "PSL K10 IID"
echo "cd src/rq2/sdar_tiny_k10_PSLiid_sglr && bash scripts/run_tiny_k10_psl_iid_sglr_all_cuts.sh"

echo
echo "PSL K10 NonIID"
echo "cd src/rq2/sdar_tiny_k10_PSLnoniid_sglr && bash scripts/run_tiny_k10_psl_noniid_sglr_all_cuts.sh"

echo
echo "PSL K10 Dirichlet"
echo "cd src/rq2/sdar_tiny_k10_PSLdirichlet_a05_sglr && bash scripts/run_tiny_k10_psl_dirichlet_a05_sglr_all_cuts.sh"
