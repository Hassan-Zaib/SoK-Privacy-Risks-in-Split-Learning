# RQ2 Multi-Client Split Learning

RQ2 studies how client count, data partitioning, and orchestration affect smashed-activation leakage.

## Environment

```bash
conda activate splitlearn
Experiments
This folder contains round-robin and parallel split-learning experiments on Tiny ImageNet.
code/sdar_tiny_k10_RRiid/
code/sdar_tiny_k10_RRnoniid/
code/sdar_tiny_k10_RRdirichlet_a05/
code/sdar_tiny_k20_RRiid/
code/sdar_tiny_k20_RRnoniid/
code/sdar_tiny_k20_RRdirichlet_a05/
code/sdar_tiny_k10_PSLiid_sglr/
code/sdar_tiny_k10_PSLnoniid_sglr/
code/sdar_tiny_k10_PSLdirichlet_a05_sglr/
Run
bash run_all.sh
Results
Final processed metrics, per-client metrics, and qualitative reconstructions are stored in results/.
