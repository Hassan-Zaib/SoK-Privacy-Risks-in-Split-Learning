# SoK Capability-Aware Security and Privacy Risks at the Split Learning Interface

This repository contains the curated artifact for the paper:

**SoK: Capability-Aware Security and Privacy Risks at the Split Learning Interface**

The artifact includes source code, environment specifications, processed metrics, and qualitative figures used in the empirical study. It does not include raw datasets, trained checkpoints, temporary files, failed runs, or smoke-test outputs.

## Repository Structure

```text
environment/  Conda environment exports for the experiments.
requirements/ Full pip package snapshots for the experiment environments.
src/          Source code for RQ1, RQ2, and RQ3.
scripts/      Top-level run helpers and command templates.
results/      Final processed metrics and qualitative figures.
docs/         Dataset setup notes, artifact notes, and result manifest.
Environments
RQ1 and RQ2 were executed with the splitlearn Conda environment.
RQ3 was executed with the rq3 Conda environment.
The exported environments are provided under:
environment/rq1_rq2_splitlearn.yml
environment/rq3.yml
Full pip snapshots are provided under:
requirements/rq1_rq2_pip_freeze.txt
requirements/rq3_pip_freeze.txt
Example setup:
conda env create -f environment/rq1_rq2_splitlearn.yml
conda env create -f environment/rq3.yml
Dataset Setup
The experiments use CIFAR-10, CIFAR-100, and Tiny ImageNet. Raw datasets are not included.
Dataset layout and Tiny ImageNet instructions are provided in:
docs/dataset_setup.md
Usage
The top-level scripts document the exact command sequence used for each research question.
bash scripts/run_rq1.sh
bash scripts/run_rq2.sh
bash scripts/run_rq3.sh
RQ1 compares representative SAIA attack families across datasets and split points under matched two-party split-learning settings.
RQ2 studies multi-client leakage under different client counts, data partitions, and orchestration strategies.
RQ3 evaluates ConvNeXt-Tiny with activation quantization and Gaussian-noise perturbations.
Results
Final processed outputs are stored under:
results/rq1/
results/rq2/
results/rq3/
The mapping between paper artifacts and repository files is documented in:
docs/result_manifest.md
Artifact Scope
This is a curated research artifact. Large checkpoints, raw logs, duplicate runs, and intermediate debugging outputs are intentionally omitted.
