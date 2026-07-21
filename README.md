# SoK Capability-Aware Security and Privacy Risks at the Split Learning Interface

This repository contains the curated artifact for the paper **SoK Capability-Aware Security and Privacy Risks at the Split Learning Interface**.

The artifact includes source code, environment specifications, processed metrics, and qualitative figures used in the empirical study. Raw datasets, checkpoints, temporary logs, failed runs, and smoke-test outputs are not included.

## Repository Layout

- `src/` contains the source code for RQ1, RQ2, and RQ3.
- `scripts/` contains top-level run helpers and per-RQ execution notes.
- `results/` contains final processed metrics and qualitative figures.
- `environment/` contains Conda environment exports.
- `requirements/` contains pip package snapshots.
- `docs/` contains dataset setup notes, artifact notes, and result mappings.

## Setup

RQ1 and RQ2 use the `splitlearn` Conda environment.

RQ3 uses the `rq3` Conda environment.

Environment files are provided under `environment/`.

Dataset setup instructions are provided in `docs/dataset_setup.md`.

## Usage

Read the per-question instructions before running experiments.

- RQ1 instructions are in `scripts/rq1/README.md`.
- RQ2 instructions are in `scripts/rq2/README.md`.
- RQ3 instructions are in `scripts/rq3/README.md`.

Top-level helper scripts are also provided.

- `bash scripts/run_rq1.sh`
- `bash scripts/run_rq2.sh`
- `bash scripts/run_rq3.sh`

These helper scripts print the command sequence used for each research question.

## Results

Final processed outputs are stored under `results/rq1/`, `results/rq2/`, and `results/rq3/`.

The mapping between paper tables, figures, and repository files is documented in `docs/result_manifest.md`.
