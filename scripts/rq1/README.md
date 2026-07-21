# RQ1 Execution Notes

RQ1 compares representative SAIA attack families under matched two-party split-learning settings.

## Environment

Activate the `splitlearn` environment.

`conda activate splitlearn`

## Dataset Setup

RQ1 uses CIFAR-10, CIFAR-100, and Tiny ImageNet. See `docs/dataset_setup.md` for dataset placement.

## Run Commands

Run commands from the repository root.

### UnSplit

`cd src/rq1/unsplit/cifar10/unsplitresnet18 && bash scripts/train_all_cuts.sh && bash scripts/attack_all_cuts.sh`

`cd src/rq1/unsplit/cifar100/unsplitresnet18-100 && bash scripts/train_all_cuts.sh && bash scripts/attack_all_cuts.sh`

`cd src/rq1/unsplit/tinyimagenet/unsplitresnet18-tiny && bash scripts/train_all_cuts.sh && bash scripts/attack_all_cuts.sh`

### SDAR

`cd src/rq1/sdar/cifar10/sdarresnet18 && bash scripts/run_sdar_resnet18_all.sh`

`cd src/rq1/sdar/cifar100/sdarresnet18-100 && bash scripts/run_sdar_resnet18_all.sh`

`cd src/rq1/sdar/tinyimagenet/sdarresnet18-tiny && bash scripts/run_sdar_resnet18_all.sh`

### GAN-DRA

`cd src/rq1/gan_dra/cifar10/maeraresnet18-10 && bash scripts/run_maera_all_cuts.sh`

`cd src/rq1/gan_dra/cifar100/maeraresnet18-100 && bash scripts/run_maera_all_cuts.sh`

`cd src/rq1/gan_dra/tinyimagenet/maeraresnet18_tiny && bash scripts/run_maera_all_cuts.sh`

### SimLIA

`cd src/rq1/simlia/cifar10 && bash scripts/train_all_cuts.sh && bash scripts/attack_all_cuts.sh`

`cd src/rq1/simlia/cifar100 && bash scripts/train_all_cuts.sh && bash scripts/attack_all_cuts.sh`

`cd src/rq1/simlia/tinyimagenet && bash scripts/train_all_cuts.sh && bash scripts/attack_all_cuts.sh`

## Outputs

Final processed RQ1 outputs are stored under `results/rq1/`.
