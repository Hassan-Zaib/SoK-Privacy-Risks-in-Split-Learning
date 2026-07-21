#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "RQ1 uses the splitlearn environment."
echo "Repository root for this RQ: $ROOT"

run_exp() {
  local dir="$1"
  shift
  echo
  echo "== Running ${dir} =="
  (
    cd "$ROOT/$dir"
    "$@"
  )
}

run_exp code/unsplit/cifar10/unsplitresnet18 bash scripts/train_all_cuts.sh
run_exp code/unsplit/cifar10/unsplitresnet18 bash scripts/attack_all_cuts.sh

run_exp code/unsplit/cifar100/unsplitresnet18-100 bash scripts/train_all_cuts.sh
run_exp code/unsplit/cifar100/unsplitresnet18-100 bash scripts/attack_all_cuts.sh

run_exp code/unsplit/tinyimagenet/unsplitresnet18-tiny bash scripts/train_all_cuts.sh
run_exp code/unsplit/tinyimagenet/unsplitresnet18-tiny bash scripts/attack_all_cuts.sh

run_exp code/sdar/cifar10/sdarresnet18 bash scripts/run_sdar_resnet18_all.sh
run_exp code/sdar/cifar100/sdarresnet18-100 bash scripts/run_sdar_resnet18_all.sh
run_exp code/sdar/tinyimagenet/sdarresnet18-tiny bash scripts/run_sdar_resnet18_all.sh

run_exp code/gan_dra/cifar10/maeraresnet18-10 bash scripts/run_maera_all_cuts.sh
run_exp code/gan_dra/cifar100/maeraresnet18-100 bash scripts/run_maera_all_cuts.sh
run_exp code/gan_dra/tinyimagenet/maeraresnet18_tiny bash scripts/run_maera_all_cuts.sh

run_exp code/simlia/cifar10 bash scripts/train_all_cuts.sh
run_exp code/simlia/cifar10 bash scripts/attack_all_cuts.sh

run_exp code/simlia/cifar100 bash scripts/train_all_cuts.sh
run_exp code/simlia/cifar100 bash scripts/attack_all_cuts.sh

run_exp code/simlia/tinyimagenet bash scripts/train_all_cuts.sh
run_exp code/simlia/tinyimagenet bash scripts/attack_all_cuts.sh
