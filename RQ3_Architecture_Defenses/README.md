# RQ3 Architecture and Activation Perturbations

RQ3 evaluates ConvNeXt-Tiny on Tiny ImageNet under activation-level perturbations.

## Environment

Use the `rq3` Conda environment.

```bash
conda env create -f ../environment/rq3.yml
conda activate rq3
Dataset
Tiny ImageNet is not included in this repository. Set DATA_ROOT to the extracted
tiny-imagenet-200 directory.
export DATA_ROOT=/path/to/tiny-imagenet-200
The expected dataset layout can be checked from the repository root with:
DATA_ROOT=/path/to/tiny-imagenet-200 bash check_datasets.sh
Artifact Check
This verifies imports, source files, and released RQ3 result files. It does not train models.
bash check_artifact.sh
Functional Smoke Test
This runs a bounded C1 no-defense test with one victim-training epoch and 20 SDAR steps. It is
intended to verify that the RQ3 implementation executes.
DATA_ROOT=/path/to/tiny-imagenet-200 \
SMOKE_ONLY=1 CUTS=c1 DEFENSES=none EPOCHS=1 SDAR_STEPS=20 \
bash run_all.sh
Full Reproduction
This launches the full RQ3 reproduction workflow across C1--C4 and the reported settings:
no defense, quantization Q4, quantization Q2, Gaussian noise 0.01, and Gaussian noise 0.05.
DATA_ROOT=/path/to/tiny-imagenet-200 bash run_all.sh
Full reproduction may take several hours depending on the GPU.
Results
Final processed metrics and qualitative reconstructions are stored in results/.
