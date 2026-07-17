# SoK Capability-Aware Security and Privacy Risks at the Split Learning Interface

This repository contains the curated artifact for the paper:

**SoK: Capability-Aware Security and Privacy Risks at the Split Learning Interface**

The artifact includes code, configurations, processed metrics, and qualitative figures used for the empirical study. It does not include raw datasets, trained checkpoints, temporary files, failed runs, or smoke-test outputs.

## Repository Structure

```text
configs/      Final experiment configuration files.
src/          Reusable implementation code.
scripts/      Commands and helper scripts for reproducing RQ1, RQ2, and RQ3 outputs.
results/      Final processed metrics and qualitative figures.
docs/         Artifact notes and result manifests.
Research Questions
RQ1
RQ1 compares representative SAIA attack families across datasets and split points under matched training conditions.
RQ2
RQ2 studies multi-client leakage under different client counts, data partitions, and orchestration strategies.
RQ3
RQ3 evaluates ConvNeXt-Tiny with activation quantization and Gaussian-noise perturbations.
Data
Raw datasets are not included. The experiments use public datasets including CIFAR-10, CIFAR-100, and Tiny ImageNet. Users should place datasets under the expected local data directory described in each script.
Results
Processed metrics and qualitative reconstruction figures are provided under results/.
Notes
This is a curated research artifact. Large checkpoints, raw logs, duplicate runs, and intermediate debugging outputs are intentionally omitted.
