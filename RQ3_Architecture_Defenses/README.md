# RQ3 Architecture and Activation Perturbations

RQ3 evaluates ConvNeXt-Tiny on Tiny ImageNet under activation-level perturbations.

## Environment

Use the `rq3` Conda environment.

```bash
conda env create -f ../environment/rq3.yml
conda activate rq3
```

## Artifact Check
The provided entrypoint verifies that the required Python packages, source files, and reported result files are present.

```bash
bash run_all.sh
```
This check does not rerun the full ConvNeXt-Tiny training workflow. Full training requires Tiny ImageNet and should be launched from the source files in `code/`.

## Settings
The reported settings are no defense, quantization Q4, quantization Q2, Gaussian noise 0.01, and Gaussian noise 0.05.

## Results
Final processed metrics and qualitative reconstructions are stored in `results/`.
