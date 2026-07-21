#!/usr/bin/env bash
set -e

echo "RQ1 uses the splitlearn environment."
echo "Run this file from RQ1_Attack_Comparison."

cd code/unsplit/cifar10/unsplitresnet18
bash scripts/train_all_cuts.sh
bash scripts/attack_all_cuts.sh
cd ../../../..

cd code/unsplit/cifar100/unsplitresnet18-100
bash scripts/train_all_cuts.sh
bash scripts/attack_all_cuts.sh
cd ../../../..

cd code/unsplit/tinyimagenet/unsplitresnet18-tiny
bash scripts/train_all_cuts.sh
bash scripts/attack_all_cuts.sh
cd ../../../..

cd code/sdar/cifar10/sdarresnet18
bash scripts/run_sdar_resnet18_all.sh
cd ../../../..

cd code/sdar/cifar100/sdarresnet18-100
bash scripts/run_sdar_resnet18_all.sh
cd ../../../..

cd code/sdar/tinyimagenet/sdarresnet18-tiny
bash scripts/run_sdar_resnet18_all.sh
cd ../../../..

cd code/gan_dra/cifar10/maeraresnet18-10
bash scripts/run_maera_all_cuts.sh
cd ../../../..

cd code/gan_dra/cifar100/maeraresnet18-100
bash scripts/run_maera_all_cuts.sh
cd ../../../..

cd code/gan_dra/tinyimagenet/maeraresnet18_tiny
bash scripts/run_maera_all_cuts.sh
cd ../../../..
