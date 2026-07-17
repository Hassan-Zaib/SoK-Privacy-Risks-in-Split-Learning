# SDAR Tiny ImageNet K=10 PSL IID SGLR

This folder implements an SGLR-style Parallel Split Learning experiment for SDAR RQ2.

## Setting

- Dataset: Tiny ImageNet
- Classes: 200
- Model: ResNet-18
- Cuts: C1, C2, C3, C4
- Clients: K=10
- Partition: IID stratified
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

Each client owns a private front model:

```text
client 0: x_0 -> f_0 -> z_0
client 1: x_1 -> f_1 -> z_1
...
client 9: x_9 -> f_9 -> z_9
```

The server owns one shared server-side model:

```text
concat(z_0, z_1, ..., z_9) -> g -> predictions
```

SGLR adds two mechanisms on top of plain PSL:

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

## Run

Single cut:

```bash
cd ~/saia_sok/sdar_tiny_k10_PSLiid_sglr
conda activate sdar
python scripts/run_sdar_psl_tiny_iid_sglr.py --cut c1
```

All cuts:

```bash
cd ~/saia_sok/sdar_tiny_k10_PSLiid_sglr
conda activate sdar
bash scripts/run_tiny_k10_psl_iid_sglr_all_cuts.sh
```

Recommended tmux:

```bash
tmux kill-session -t sdar_tiny_k10_PSLiid_sglr 2>/dev/null || true

tmux new-session -d -s sdar_tiny_k10_PSLiid_sglr \
"bash -lc 'source ~/miniconda/etc/profile.d/conda.sh && conda activate sdar && cd ~/saia_sok/sdar_tiny_k10_PSLiid_sglr && mkdir -p data && ln -sfn ~/saia_sok/datasets/tiny-imagenet-200 data/tiny-imagenet-200 && rm -rf results && bash scripts/run_tiny_k10_psl_iid_sglr_all_cuts.sh; echo; echo \"[sdar tiny k10 psl iid sglr all cuts finished]\"; exec bash'"
```

## Outputs

Each cut writes to:

```text
results/sdar_psl_iid_sglr/tinyimagenet_resnet18_c*_K10_iid_psl_sglr_run1/
```

Important files:

- `summary.json`: global reconstruction metrics, fixed-probe metrics, train/test accuracy, timing, communication, SplitLr/SplitAvg settings
- `per_client_metrics.json`: MSE/PSNR/SSIM/LPIPS/ASR per client
- `test_accuracy.json`: held-out Tiny ImageNet validation accuracy per client
- `fixed_probe_grid.png`: fixed original/reconstruction grid
- `fixed_probe_originals.npy`, `fixed_probe_reconstructions.npy`: fixed fair-comparison probe arrays
- `weights/front_*.weights.h5`, `weights/server_g.weights.h5`: saved trained split-model weights

## Comparison

Use this as the practical aligned-PSL middle point between:

- `sdar_tiny_k10_PSLiid`: plain separate-front PSL, no SplitAvg/SplitLr
- `sdar_tiny_k10_PSLiid_sharedfront`: fully shared-front PSL control
- `sdar_tiny_k10_RRiid`: SDAR-style round-robin shared-front baseline
