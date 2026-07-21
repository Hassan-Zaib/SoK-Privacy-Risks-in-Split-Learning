#!/usr/bin/env bash
set -euo pipefail

echo "Checking shell scripts"
bash -n RQ1_Attack_Comparison/run_all.sh
bash -n RQ2_Multi_Client/run_all.sh
bash -n RQ3_Architecture_Defenses/run_all.sh
find RQ1_Attack_Comparison RQ2_Multi_Client -type f -name "*.sh" -print0 | xargs -0 -n1 bash -n

echo "Checking RQ2 data loaders"
count=$(find RQ2_Multi_Client/code -path "*/src/data.py" | wc -l)
test "$count" -eq 9

echo "Checking RQ3 artifact files"
(
  cd RQ3_Architecture_Defenses
  bash check_artifact.sh
)

echo "Artifact check passed"
