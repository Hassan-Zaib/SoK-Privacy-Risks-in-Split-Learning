# RQ3 Execution Notes

RQ3 evaluates ConvNeXt-Tiny on Tiny ImageNet under activation-level perturbations.

## Environment

Activate the `rq3` environment.

`conda activate rq3`

## Dataset Setup

RQ3 uses Tiny ImageNet. See `docs/dataset_setup.md` for dataset placement.

## Source Code

The reusable ConvNeXt-Tiny implementation is stored under `src/rq3/`.

The main source files are:

- `src/rq3/convnext_split.py`
- `src/rq3/data_tinyimagenet.py`
- `src/rq3/full_sdar.py`
- `src/rq3/perturbations.py`
- `src/rq3/train_utils.py`

## Reported Settings

The reported settings are:

- No defense
- Quantization Q4
- Quantization Q2
- Noise 0.01
- Noise 0.05

## Outputs

Final processed RQ3 outputs are stored under `results/rq3/`.

Metric summaries are under `results/rq3/summaries/`.

Qualitative figures are under `results/rq3/figures/`.
