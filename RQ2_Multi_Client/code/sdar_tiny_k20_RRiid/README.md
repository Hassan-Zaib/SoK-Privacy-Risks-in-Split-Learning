# SDAR Tiny ImageNet K=10 Round-Robin IID

This folder is the IID counterpart of `sdar_tiny_k10_RRnoniid` for RQ2.

## Setting

- Dataset: Tiny ImageNet
- Classes: 200
- Model: ResNet-18
- Cuts: C1, C2, C3, C4
- Clients: K=10
- Partition: IID balanced split
- Orchestration: sequential round-robin
- Attack: SDAR
- Batch size: 128
- Iterations: 20,000

In this IID setting, every client receives samples from all 200 Tiny ImageNet classes. The split is
balanced per class across the 10 clients, so each client has approximately the same number of images
from every class. The server-side auxiliary pool is split in the same IID-balanced style.

## Data Path

The TensorFlow SDAR loader expects Tiny ImageNet here:

```text
data/tiny-imagenet-200/
```

On the server, if the dataset already exists at `~/saia_sok/datasets/tiny-imagenet-200`, create a symlink:

```bash
cd ~/saia_sok/sdar_tiny_k10_RRiid
mkdir -p data
ln -sfn ~/saia_sok/datasets/tiny-imagenet-200 data/tiny-imagenet-200
ls data/tiny-imagenet-200/wnids.txt
```

## Run

Single cut:

```bash
cd ~/saia_sok/sdar_tiny_k10_RRiid
conda activate sdar
python scripts/run_sdar_rr_tiny_iid.py --cut c1
```

All cuts:

```bash
cd ~/saia_sok/sdar_tiny_k10_RRiid
conda activate sdar
bash scripts/run_tiny_k10_rr_iid_all_cuts.sh
```

Recommended tmux:

```bash
tmux new-session -d -s sdar_tiny_k10_RRiid \
"bash -lc 'source ~/miniconda/etc/profile.d/conda.sh && conda activate sdar && cd ~/saia_sok/sdar_tiny_k10_RRiid && bash scripts/run_tiny_k10_rr_iid_all_cuts.sh; echo; echo \"[sdar tiny k10 rr iid finished]\"; exec bash'"
```

## Outputs

Each cut writes to:

```text
results/sdar_rr_iid/tinyimagenet_resnet18_c*_K10_iid_rr_run1/
```

Files include:

- `summary.json`: global metrics, best/worst client SSIM, timing, communication size
- `per_client_metrics.json`: MSE/PSNR/SSIM/LPIPS/ASR for each client
- `communication.json`: smashed-data tensor shape and byte counts
- `timing.json`: runtime and per-round timing
- `original_recon_grid.png`: aggregate visual comparison
- `client_0_grid.png` ... `client_9_grid.png`: per-client visual comparisons
- `originals.npy`, `reconstructions.npy`, `labels.npy`
