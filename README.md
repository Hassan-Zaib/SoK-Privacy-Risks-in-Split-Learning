# SoK Capability-Aware Security and Privacy Risks at the Split Learning Interface

This repository contains the code, configurations, processed results, and plotting scripts for the experiments reported in the paper.

## Scope

The artifact covers three experimental questions.

- RQ1 compares representative SAIA attack families across datasets and cut points.
- RQ2 studies multi-client leakage under different client counts, data partitions, and orchestration strategies.
- RQ3 evaluates architecture and activation-level perturbations using ConvNeXt-Tiny.

## Repository Layout

- `src/` contains reusable implementation code.
- `configs/` contains experiment configurations.
- `scripts/` contains commands for running experiments and regenerating figures.
- `results/` contains processed metrics, summaries, and final figures.
- `docs/` contains the result manifest and artifact notes.

## Data

Datasets are not included. Place datasets under a local `data/` directory when running the code.

## Results

The final processed results used in the paper are stored under `results/`.
The file `docs/result_manifest.md` maps paper tables and figures to their corresponding files.

## Status

This repository is being prepared as the artifact release for the paper.
