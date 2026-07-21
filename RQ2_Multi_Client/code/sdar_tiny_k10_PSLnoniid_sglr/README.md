# SDAR Tiny ImageNet K=10 PSL non-IID SGLR

This folder implements the label-shard non-IID counterpart of `sdar_tiny_k10_PSLiid_sglr`.

## Setting

- Dataset: Tiny ImageNet
- Classes: 200
- Model: ResNet-18
- Cuts: C1, C2, C3, C4
- Clients: K=10
- Partition: label-shard non-IID by `label % K`
- Classes per client: 20
- Client front models: separate `f_i`
- Server model: one shared `g`
- Orchestration: synchronous PSL with SGLR-style SplitLr and SplitAvg
- Client batch size: 32
- Effective server batch size per global round: 320
- Default SplitLr alpha: 0.5
- Default SplitAvg phi: 0.5
- Attack: SDAR with one shared simulator and decoder
- Iterations: 20,000 global rounds

## Architecture

Each client owns a private front model and one label shard:

```text
client 0: classes 0, 10, 20, ... -> f_0 -> z_0
client 1: classes 1, 11, 21, ... -> f_1 -> z_1
...
client 9: classes 9, 19, 29, ... -> f_9 -> z_9
```

The server owns one shared server-side model:

```text
concat(z_0, z_1, ..., z_9) -> g -> predictions
```

SGLR adds two mechanisms on top of plain separate-front PSL:

```text
SplitLr:
    client_lr = base_lr
    server_lr = base_lr * K^alpha

SplitAvg:
    sample active clients C_a
    dz_avg = mean(dz_i for i in C_a)
    active clients receive dz_avg
    non-active clients receive their own dz_i
```

With the default K=10 and phi=0.5, five clients are active for SplitAvg in each round.

## Accuracy Reporting

This non-IID experiment saves two accuracy files:

- `local_test_accuracy.json`: each client is tested only on its own 20-class label shard.
- `global_test_accuracy.json`: each client front is tested on the full 200-class Tiny ImageNet validation set.

Use the global metric when comparing against IID experiments. The local shard metric is useful but easier because each client sees only its own label subset.

## Run

Single cut:

```bash
cd ~/saia_sok/sdar_tiny_k10_PSLnoniid_sglr
conda activate sdar
python scripts/run_sdar_psl_tiny_noniid_sglr.py --cut c1
```

All cuts:

```bash
cd ~/saia_sok/sdar_tiny_k10_PSLnoniid_sglr
conda activate sdar
bash scripts/run_tiny_k10_psl_noniid_sglr_all_cuts.sh
```

Recommended tmux:

```bash
tmux kill-session -t sdar_tiny_k10_PSLnoniid_sglr 2>/dev/null || true

tmux new-session -d -s sdar_tiny_k10_PSLnoniid_sglr \
"bash -lc 'source ~/miniconda/etc/profile.d/conda.sh && conda activate sdar && cd ~/saia_sok/sdar_tiny_k10_PSLnoniid_sglr && mkdir -p data && ln -sfn ~/saia_sok/datasets/tiny-imagenet-200 data/tiny-imagenet-200 && rm -rf results && bash scripts/run_tiny_k10_psl_noniid_sglr_all_cuts.sh; echo; echo \"[sdar tiny k10 psl noniid sglr all cuts finished]\"; exec bash'"
```

## Outputs

Each cut writes to:

```text
results/sdar_psl_noniid_sglr/tinyimagenet_resnet18_c*_K10_noniid_psl_sglr_run1/
```

Important files:

- `summary.json`: global reconstruction metrics, fixed-probe metrics, local/global test accuracy, timing, communication, SplitLr/SplitAvg settings
- `per_client_metrics.json`: MSE/PSNR/SSIM/LPIPS/ASR per client
- `local_test_accuracy.json`: held-out label-shard test accuracy per client
- `global_test_accuracy.json`: full 200-class test accuracy per client
- `fixed_probe_grid.png`: fixed original/reconstruction grid
- `fixed_probe_originals.npy`, `fixed_probe_reconstructions.npy`: fixed fair-comparison probe arrays
- `weights/front_*.weights.h5`, `weights/server_g.weights.h5`: saved trained split-model weights

## Comparison

Use this experiment as the aligned-gradient non-IID separate-front point between:

- `sdar_tiny_k10_PSLnoniid`: plain separate-front PSL, no SplitAvg/SplitLr
- `sdar_tiny_k10_PSLnoniid_sharedfront`: fully shared-front PSL control
- `sdar_tiny_k10_RRnoniid`: SDAR-style round-robin shared-front baseline
