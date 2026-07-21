#!/usr/bin/env bash
set -euo pipefail

source ~/miniconda/etc/profile.d/conda.sh
conda activate splitlearn

echo "RQ1 uses the splitlearn environment."
echo "Run the following commands from the repository root."

echo
echo "UnSplit CIFAR-10"
echo "cd src/rq1/unsplit/cifar10/unsplitresnet18 && bash scripts/train_all_cuts.sh && bash scripts/attack_all_cuts.sh"

echo
echo "UnSplit CIFAR-100"
echo "cd src/rq1/unsplit/cifar100/unsplitresnet18-100 && bash scripts/train_all_cuts.sh && bash scripts/attack_all_cuts.sh"

echo
echo "UnSplit Tiny ImageNet"
echo "cd src/rq1/unsplit/tinyimagenet/unsplitresnet18-tiny && bash scripts/train_all_cuts.sh && bash scripts/attack_all_cuts.sh"

echo
echo "SDAR CIFAR-10"
echo "cd src/rq1/sdar/cifar10/sdarresnet18 && bash scripts/run_sdar_resnet18_all.sh"

echo
echo "SDAR CIFAR-100"
echo "cd src/rq1/sdar/cifar100/sdarresnet18-100 && bash scripts/run_sdar_resnet18_all.sh"

echo
echo "SDAR Tiny ImageNet"
echo "cd src/rq1/sdar/tinyimagenet/sdarresnet18-tiny && bash scripts/run_sdar_resnet18_all.sh"

echo
echo "GAN-DRA CIFAR-10"
echo "cd src/rq1/gan_dra/cifar10/maeraresnet18-10 && bash scripts/run_maera_all_cuts.sh"

echo
echo "GAN-DRA CIFAR-100"
echo "cd src/rq1/gan_dra/cifar100/maeraresnet18-100 && bash scripts/run_maera_all_cuts.sh"

echo
echo "GAN-DRA Tiny ImageNet"
echo "cd src/rq1/gan_dra/tinyimagenet/maeraresnet18_tiny && bash scripts/run_maera_all_cuts.sh"

echo
echo "SimLIA CIFAR-10"
echo "cd src/rq1/simlia/cifar10 && bash scripts/train_all_cuts.sh && bash scripts/attack_all_cuts.sh"

echo
echo "SimLIA CIFAR-100"
echo "cd src/rq1/simlia/cifar100 && bash scripts/train_all_cuts.sh && bash scripts/attack_all_cuts.sh"

echo
echo "SimLIA Tiny ImageNet"
echo "cd src/rq1/simlia/tinyimagenet && bash scripts/train_all_cuts.sh && bash scripts/attack_all_cuts.sh"
