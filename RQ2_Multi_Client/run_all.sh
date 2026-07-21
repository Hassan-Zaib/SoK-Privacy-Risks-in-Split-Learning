#!/usr/bin/env bash
set -e

echo "RQ2 uses the splitlearn environment."
echo "Run this file from RQ2_Multi_Client."

cd code/sdar_tiny_k10_RRiid
bash scripts/run_tiny_k10_rr_iid_all_cuts.sh
cd ../..

cd code/sdar_tiny_k10_RRnoniid
bash scripts/run_tiny_k10_rr_noniid_all_cuts.sh
cd ../..

cd code/sdar_tiny_k10_RRdirichlet_a05
bash scripts/run_tiny_k10_rr_dirichlet_a05_all_cuts.sh
cd ../..

cd code/sdar_tiny_k20_RRiid
bash scripts/run_tiny_k20_rr_iid_all_cuts.sh
cd ../..

cd code/sdar_tiny_k20_RRnoniid
bash scripts/run_tiny_k20_rr_noniid_all_cuts.sh
cd ../..

cd code/sdar_tiny_k20_RRdirichlet_a05
bash scripts/run_tiny_k20_rr_dirichlet_a05_all_cuts.sh
cd ../..

cd code/sdar_tiny_k10_PSLiid_sglr
bash scripts/run_tiny_k10_psl_iid_sglr_all_cuts.sh
cd ../..

cd code/sdar_tiny_k10_PSLnoniid_sglr
bash scripts/run_tiny_k10_psl_noniid_sglr_all_cuts.sh
cd ../..

cd code/sdar_tiny_k10_PSLdirichlet_a05_sglr
bash scripts/run_tiny_k10_psl_dirichlet_a05_sglr_all_cuts.sh
cd ../..
