# RQ2 Multi-Client Split Learning

RQ2 studies how client count, data partitioning, and orchestration affect smashed-activation leakage.

Environment

Use the sdar Conda environment.

Run

```bash
conda activate sdar
bash run_all.sh
```

## Full Reproduction

`run_all.sh` launches the full RQ2 multi-client experiment set. It may take several hours and requires Tiny ImageNet as described in `../docs/dataset_setup.md`.

Code Layout

The code folder contains round-robin and parallel split-learning experiments on Tiny ImageNet.

Results

Final processed metrics, per-client metrics, and qualitative reconstructions are stored in results.
