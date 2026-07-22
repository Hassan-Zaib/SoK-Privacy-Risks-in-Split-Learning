# SoK Capability-Aware Security and Privacy Risks at the Split Learning Interface

This repository contains the artifact for the paper **SoK Capability-Aware Security and Privacy Risks at the Split Learning Interface**.

The artifact is organized by research question. Each folder contains the code, run instructions, and final processed results for that part of the empirical study. Raw datasets, checkpoints, temporary logs, failed runs, and smoke-test outputs are not included.

## Repository Structure

- `RQ1_Attack_Comparison/` contains the two-party attack comparison across datasets and cut points.
- `RQ2_Multi_Client/` contains the multi-client split learning experiments.
- `RQ3_Architecture_Defenses/` contains the ConvNeXt-Tiny and activation perturbation experiments.
- `environment/` contains Conda environment exports.
- `requirements/` contains pip package snapshots.
- `docs/` contains dataset setup notes, artifact notes, and result mappings.

## Setup

RQ1 uses the `splitlearn` Conda environment, RQ2 uses the `sdar` Conda environment, and RQ3 uses the `rq3` Conda environment.

Run `conda env create -f environment/rq1_splitlearn.yml` for RQ1, `conda env create -f environment/rq2_sdar.yml` for RQ2, and `conda env create -f environment/rq3.yml` for RQ3.


Dataset setup instructions are provided in `docs/dataset_setup.md`.

## Usage

For RQ1, run `cd RQ1_Attack_Comparison` and then `bash run_all.sh`.

For RQ2, run `cd RQ2_Multi_Client` and then `bash run_all.sh`.

For RQ3, run `cd RQ3_Architecture_Defenses` and then `bash run_all.sh`.

The mapping between paper tables, figures, and artifact files is documented in `docs/result_manifest.md`.
## Quick Reviewer Validation

These representative smoke tests verify that the main execution path of each research question works and produces the expected output files. Smoke-test metrics are not intended to reproduce the final paper results.

### RQ1: MAERA, CIFAR-10, ResNet-18, C1

```bash
conda run --no-capture-output -n splitlearn \
  bash -c 'cd RQ1_Attack_Comparison/code/gan_dra/cifar10/maeraresnet18-10 && bash scripts/run_maera_smoke.sh'
```

Expected output:

```text
RQ1_Attack_Comparison/code/gan_dra/cifar10/maeraresnet18-10/results/maera/cifar10_resnet18_c1/summary.json
```

### RQ2: SDAR, K=10 Round-Robin IID, C1

```bash
cd RQ2_Multi_Client/code/sdar_tiny_k10_RRiid

TINY_IMAGENET_ROOT=/path/to/tiny-imagenet-200 \
PYTHONUNBUFFERED=1 \
conda run --no-capture-output -n sdar \
python -u scripts/run_sdar_rr_tiny_iid.py \
  --cut c1 \
  --num-clients 10 \
  --batch-size 4 \
  --num-iter 10 \
  --eval-batches 1 \
  --output-dir results/reviewer_smoke \
  --print-to-stdout
```

Expected output:

```text
results/reviewer_smoke/tinyimagenet_resnet18_c1_K10_iid_rr_run1/summary.json
```

### RQ3: ConvNeXt-Tiny and Full SDAR, C1

```bash
cd RQ3_Architecture_Defenses

DATA_ROOT=/path/to/tiny-imagenet-200 \
CUTS=c1 \
BATCH_SIZE=256 \
NUM_WORKERS=4 \
SDAR_BATCH_SIZE=8 \
SMOKE_ONLY=1 \
PYTHONUNBUFFERED=1 \
conda run --no-capture-output -n rq3 \
bash scripts/run_rq3_sequential.sh
```

Expected outputs:

```text
results_smoke/none/c1/victim/victim_summary.json
results_smoke/none/c1/sdar/full_sdar_summary.json
```

Full reproduction commands are provided separately in the research-question directories. Full experiments require substantially more computation than the representative smoke tests.
