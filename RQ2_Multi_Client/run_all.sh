#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "RQ2 uses the sdar environment."
echo "Repository root for this RQ: $ROOT"

run_exp() {
  local dir="$1"
  local script="$2"
  echo
  echo "== Running ${dir} =="
  (
    cd "$ROOT/$dir"
    bash "$script"
  )
}

run_exp code/sdar_tiny_k10_RRiid scripts/run_tiny_k10_rr_iid_all_cuts.sh
run_exp code/sdar_tiny_k10_RRnoniid scripts/run_tiny_k10_rr_noniid_all_cuts.sh
run_exp code/sdar_tiny_k10_RRdirichlet_a05 scripts/run_tiny_k10_rr_dirichlet_a05_all_cuts.sh
run_exp code/sdar_tiny_k20_RRiid scripts/run_tiny_k20_rr_iid_all_cuts.sh
run_exp code/sdar_tiny_k20_RRnoniid scripts/run_tiny_k20_rr_noniid_all_cuts.sh
run_exp code/sdar_tiny_k20_RRdirichlet_a05 scripts/run_tiny_k20_rr_dirichlet_a05_all_cuts.sh
run_exp code/sdar_tiny_k10_PSLiid_sglr scripts/run_tiny_k10_psl_iid_sglr_all_cuts.sh
run_exp code/sdar_tiny_k10_PSLnoniid_sglr scripts/run_tiny_k10_psl_noniid_sglr_all_cuts.sh
run_exp code/sdar_tiny_k10_PSLdirichlet_a05_sglr scripts/run_tiny_k10_psl_dirichlet_a05_sglr_all_cuts.sh
