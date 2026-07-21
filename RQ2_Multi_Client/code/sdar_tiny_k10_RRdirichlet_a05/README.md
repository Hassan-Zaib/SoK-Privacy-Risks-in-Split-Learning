# SDAR Tiny ImageNet K=10 Round-Robin Dirichlet Non-IID Alpha 0.5

This folder is the realistic non-IID Tiny ImageNet counterpart of `sdar_tiny_k10_RRiid` for RQ2.

## Setting

- Dataset: Tiny ImageNet
- Classes: 200
- Model: ResNet-18
- Cuts: C1, C2, C3, C4
- Clients: K=10
- Partition: Dirichlet non-IID, alpha=0.5
- Orchestration: sequential round-robin
- Attack: SDAR
- Batch size: 128
- Iterations: 20,000

Dirichlet alpha=0.5 gives each client a mixed but skewed class distribution. Unlike label-shard
non-IID, clients can contain many/all classes, but the class proportions differ across clients.

Following the SDAR data-loading style, Tiny ImageNet train and validation images are concatenated
and split into client/private and server/auxiliary pools. Both pools are then partitioned independently
with Dirichlet alpha=0.5 so the heterogeneous SDAR attacker interface remains unchanged.

## Data Path

The TensorFlow SDAR loader expects Tiny ImageNet here:

```text
data/tiny-imagenet-200/
```

On the server, if the dataset already exists at `~/saia_sok/datasets/tiny-imagenet-200`, create a symlink:

```bash
cd ~/saia_sok/sdar_tiny_k10_RRdirichlet_a05
mkdir -p data
ln -sfn ~/saia_sok/datasets/tiny-imagenet-200 data/tiny-imagenet-200
ls data/tiny-imagenet-200/wnids.txt
```

## Run

Single cut:

```bash
cd ~/saia_sok/sdar_tiny_k10_RRdirichlet_a05
conda activate sdar
python scripts/run_sdar_rr_tiny_dirichlet.py --cut c1 --alpha 0.5
```

All cuts:

```bash
cd ~/saia_sok/sdar_tiny_k10_RRdirichlet_a05
conda activate sdar
bash scripts/run_tiny_k10_rr_dirichlet_a05_all_cuts.sh
```

Recommended tmux:

```bash
tmux new-session -d -s sdar_tiny_k10_RRdirichlet_a05 \
"bash -lc 'source ~/miniconda/etc/profile.d/conda.sh && conda activate sdar && cd ~/saia_sok/sdar_tiny_k10_RRdirichlet_a05 && bash scripts/run_tiny_k10_rr_dirichlet_a05_all_cuts.sh; echo; echo \"[sdar tiny k10 rr dirichlet a05 finished]\"; exec bash'"
```

## Outputs

Each cut writes to:

```text
results/sdar_rr_dirichlet_a05/tinyimagenet_resnet18_c*_K10_dirichlet_a0p5_rr_run1/
```

Files include:

- `summary.json`: global metrics, best/worst client SSIM, timing, communication size
- `per_client_metrics.json`: MSE/PSNR/SSIM/LPIPS/ASR for each client
- `communication.json`: smashed-data tensor shape and byte counts
- `timing.json`: runtime and per-round timing
- `original_recon_grid.png`: aggregate visual comparison
- `client_0_grid.png` ... `client_9_grid.png`: per-client visual comparisons
- `originals.npy`, `reconstructions.npy`, `labels.npy`
