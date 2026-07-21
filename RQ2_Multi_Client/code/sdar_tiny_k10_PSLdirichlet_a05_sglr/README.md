# SDAR Tiny ImageNet K=10 PSL Dirichlet a=0.5 SGLR

This folder implements the Dirichlet alpha=0.5 counterpart of `sdar_tiny_k10_PSLiid_sglr` and `sdar_tiny_k10_PSLnoniid_sglr`.

## Setting

- Dataset: Tiny ImageNet
- Classes: 200
- Model: ResNet-18
- Cuts: C1, C2, C3, C4
- Clients: K=10
- Partition: Dirichlet non-IID, alpha=0.5
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
client 0: Dirichlet subset -> f_0 -> z_0
client 1: Dirichlet subset -> f_1 -> z_1
...
client 9: Dirichlet subset -> f_9 -> z_9
```

The server owns one shared server-side model:

```text
z_i -> g -> prediction
```

SGLR keeps client weights separate. It does not FedAvg the client fronts. It only applies SplitLr and SplitAvg at the split-gradient interface:

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

With K=10 and phi=0.5, five clients are active for SplitAvg in each round.

## Accuracy Reporting

This experiment saves both local and global utility:

- `local_test_accuracy.json`: each client is tested on its deterministic Dirichlet validation subset.
- `global_test_accuracy.json`: each client front is tested on the full 200-class Tiny ImageNet validation set.

The global metric is the stricter metric for a shared global model. The local metric is useful for client-personalized utility.

## Run

Single cut:

```bash
cd ~/saia_sok/sdar_tiny_k10_PSLdirichlet_a05_sglr
conda activate sdar
python scripts/run_sdar_psl_tiny_dirichlet_a05_sglr.py --cut c1
```

All cuts:

```bash
cd ~/saia_sok/sdar_tiny_k10_PSLdirichlet_a05_sglr
conda activate sdar
bash scripts/run_tiny_k10_psl_dirichlet_a05_sglr_all_cuts.sh
```

Recommended tmux:

```bash
tmux kill-session -t sdar_tiny_k10_PSLdirichlet_a05_sglr 2>/dev/null || true

tmux new-session -d -s sdar_tiny_k10_PSLdirichlet_a05_sglr \
"bash -lc 'source ~/miniconda/etc/profile.d/conda.sh && conda activate sdar && cd ~/saia_sok/sdar_tiny_k10_PSLdirichlet_a05_sglr && mkdir -p data && ln -sfn ~/saia_sok/datasets/tiny-imagenet-200 data/tiny-imagenet-200 && rm -rf results && bash scripts/run_tiny_k10_psl_dirichlet_a05_sglr_all_cuts.sh; echo; echo \"[sdar tiny k10 psl dirichlet a05 sglr all cuts finished]\"; exec bash'"
```

## Outputs

Each cut writes to:

```text
results/sdar_psl_dirichlet_a05_sglr/tinyimagenet_resnet18_c*_K10_dirichlet_a0p5_psl_sglr_run1/
```

Important files:

- `summary.json`: reconstruction metrics, fixed-probe metrics, local/global test accuracy, timing, communication, SplitLr/SplitAvg settings
- `per_client_metrics.json`: MSE/PSNR/SSIM/LPIPS/ASR per client
- `local_test_accuracy.json`: Dirichlet local validation accuracy per client
- `global_test_accuracy.json`: full 200-class test accuracy per client
- `fixed_probe_grid.png`: fixed original/reconstruction grid
- `fixed_probe_originals.npy`, `fixed_probe_reconstructions.npy`: fixed fair-comparison probe arrays
- `weights/front_*.weights.h5`, `weights/server_g.weights.h5`: saved trained split-model weights
