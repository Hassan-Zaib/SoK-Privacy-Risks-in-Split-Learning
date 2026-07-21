# SDAR Tiny ImageNet K=10 Round-Robin Non-IID

This folder is the Tiny ImageNet RQ2 multi-client setting using the SDAR-aligned label-modulo non-IID partition.

## Setting

- Dataset: Tiny ImageNet
- Classes: 200
- Model: ResNet-18
- Cuts: C1, C2, C3, C4
- Clients: K=10
- Partition: non-IID label skew
- Classes/client: 20
- Orchestration: sequential round-robin
- Attack: SDAR
- Batch size: 128
- Iterations: 20,000

The non-IID split follows the official SDAR hetero-client code style: labels are assigned by `label % K`.
For K=10 on Tiny ImageNet, each client receives 20 labels:

- client 0: labels 0, 10, 20, ..., 190
- client 1: labels 1, 11, 21, ..., 191
- client 2: labels 2, 12, 22, ..., 192
- ...
- client 9: labels 9, 19, 29, ..., 199

The attacker auxiliary data is split with the same rule, matching SDAR's paper/code assumption that the
server knows which client a batch came from and can curate client-specific auxiliary data.

## Data Path

The TensorFlow SDAR loader expects Tiny ImageNet here:

```text
data/tiny-imagenet-200/
```

On the server, if the dataset already exists at `~/saia_sok/datasets/tiny-imagenet-200`, create a symlink:

```bash
cd ~/saia_sok/sdar_tiny_k10_RRnoniid
mkdir -p data
ln -sfn ~/saia_sok/datasets/tiny-imagenet-200 data/tiny-imagenet-200
ls data/tiny-imagenet-200/wnids.txt
```

## Run

Single cut:

```bash
cd ~/saia_sok/sdar_tiny_k10_RRnoniid
conda activate sdar
python scripts/run_sdar_rr_tiny_noniid.py --cut c1
```

All cuts:

```bash
cd ~/saia_sok/sdar_tiny_k10_RRnoniid
conda activate sdar
bash scripts/run_tiny_k10_rr_noniid_all_cuts.sh
```

Recommended tmux:

```bash
tmux new-session -d -s sdar_tiny_k10_RRnoniid \
"bash -lc 'source ~/miniconda/etc/profile.d/conda.sh && conda activate sdar && cd ~/saia_sok/sdar_tiny_k10_RRnoniid && bash scripts/run_tiny_k10_rr_noniid_all_cuts.sh; echo; echo \"[sdar tiny k10 rr noniid finished]\"; exec bash'"
```

## Outputs

Each cut writes to:

```text
results/sdar_rr_noniid/tinyimagenet_resnet18_c*_K10_noniid_rr_run1/
```

Files include:

- `summary.json`: global metrics, best/worst client SSIM, timing, communication size
- `per_client_metrics.json`: MSE/PSNR/SSIM/LPIPS/ASR for each client
- `communication.json`: smashed-data tensor shape and byte counts
- `timing.json`: runtime and per-round timing
- `original_recon_grid.png`: aggregate visual comparison
- `client_0_grid.png` ... `client_9_grid.png`: per-client visual comparisons
- `originals.npy`, `reconstructions.npy`, `labels.npy`
