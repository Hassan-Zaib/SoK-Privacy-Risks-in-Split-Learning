# SDAR Tiny ImageNet K=20 Round-Robin Non-IID

This folder is an RQ2 multi-client adaptation of SDAR for Tiny ImageNet with ResNet-18 C1-C4 cuts.

Changed from `sdar_tiny_k10_RRnoniid`:

- Clients: K=20
- Partition: label-shard non-IID by modulo class assignment
- Classes/client: 10 because Tiny ImageNet has 200 classes
- Orchestration: round-robin
- Decoder target: Tiny ImageNet native 64x64 output

Kept the same:

- Dataset: Tiny ImageNet
- Model: ResNet-18
- Cuts: C1-C4
- Batch size: 128
- SDAR config and iterations
- Metrics: global + per-client + communication/runtime

Run all cuts:

```bash
cd ~/saia_sok/sdar_tiny_k20_RRnoniid
bash scripts/run_tiny_k20_rr_noniid_all_cuts.sh
```

Expected output folders:

```text
results/sdar_rr_noniid/tinyimagenet_resnet18_c*_K20_noniid_rr_run1/
```
