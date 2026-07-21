# SoK Capability-Aware Security and Privacy Risks at the Split Learning Interface

This repository contains the artifact for the paper **SoK Capability-Aware Security and Privacy Risks at the Split Learning Interface**.

The artifact is organized by research question. Each folder contains the code, run instructions, and final processed results for that part of the empirical study. Raw datasets, checkpoints, temporary logs, failed runs, and smoke-test outputs are not included.

## Structure

```text
RQ1_Attack_Comparison/          Two-party attack comparison across datasets and cuts
RQ2_Multi_Client/               Multi-client split learning experiments
RQ3_Architecture_Defenses/      ConvNeXt-Tiny and activation perturbation experiments
environment/                    Conda environment exports
requirements/                   Pip package snapshots
docs/                           Dataset setup, artifact notes, and result manifest
Setup
RQ1 and RQ2 use the splitlearn Conda environment.
conda env create -f environment/rq1_rq2_splitlearn.yml
conda activate splitlearn
RQ3 uses the rq3 Conda environment.
conda env create -f environment/rq3.yml
conda activate rq3
Dataset setup instructions are provided in docs/dataset_setup.md.
Usage
cd RQ1_Attack_Comparison
bash run_all.sh
cd RQ2_Multi_Client
bash run_all.sh
cd RQ3_Architecture_Defenses
bash run_all.sh
```text
