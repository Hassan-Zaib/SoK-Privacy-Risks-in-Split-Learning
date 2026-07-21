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

Run `conda env create -f environment/rq1_rq2_splitlearn.yml` for RQ1, `conda env create -f environment/rq2_sdar.yml` for RQ2, and `conda env create -f environment/rq3.yml` for RQ3.


Dataset setup instructions are provided in `docs/dataset_setup.md`.

## Usage

For RQ1, run `cd RQ1_Attack_Comparison` and then `bash run_all.sh`.

For RQ2, run `cd RQ2_Multi_Client` and then `bash run_all.sh`.

For RQ3, run `cd RQ3_Architecture_Defenses` and then `bash run_all.sh`.

The mapping between paper tables, figures, and artifact files is documented in `docs/result_manifest.md`.
