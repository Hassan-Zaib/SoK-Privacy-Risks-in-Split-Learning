# RQ2 Execution Notes

RQ2 evaluates multi-client leakage on Tiny ImageNet using SDAR.

## Environment

Activate the `splitlearn` environment.

`conda activate splitlearn`

## Dataset Setup

RQ2 uses Tiny ImageNet. See `docs/dataset_setup.md` for dataset placement.

## Run Commands

Run commands from the repository root.

### Round-Robin Settings

`cd src/rq2/sdar_tiny_k10_RRiid && bash scripts/run_tiny_k10_rr_iid_all_cuts.sh`

`cd src/rq2/sdar_tiny_k10_RRnoniid && bash scripts/run_tiny_k10_rr_noniid_all_cuts.sh`

`cd src/rq2/sdar_tiny_k10_RRdirichlet_a05 && bash scripts/run_tiny_k10_rr_dirichlet_a05_all_cuts.sh`

`cd src/rq2/sdar_tiny_k20_RRiid && bash scripts/run_tiny_k20_rr_iid_all_cuts.sh`

`cd src/rq2/sdar_tiny_k20_RRnoniid && bash scripts/run_tiny_k20_rr_noniid_all_cuts.sh`

`cd src/rq2/sdar_tiny_k20_RRdirichlet_a05 && bash scripts/run_tiny_k20_rr_dirichlet_a05_all_cuts.sh`

### PSL Settings

`cd src/rq2/sdar_tiny_k10_PSLiid_sglr && bash scripts/run_tiny_k10_psl_iid_sglr_all_cuts.sh`

`cd src/rq2/sdar_tiny_k10_PSLnoniid_sglr && bash scripts/run_tiny_k10_psl_noniid_sglr_all_cuts.sh`

`cd src/rq2/sdar_tiny_k10_PSLdirichlet_a05_sglr && bash scripts/run_tiny_k10_psl_dirichlet_a05_sglr_all_cuts.sh`

## Outputs

Final processed RQ2 outputs are stored under `results/rq2/`.
