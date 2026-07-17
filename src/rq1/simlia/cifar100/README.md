# Sim-LIA ResNet-18 CIFAR-100

This folder adapts the official Sim-LIA repository to the SAIA benchmark.

Official code is copied under:

```text
official_simlia/
```

The benchmark wrapper keeps the Sim-LIA smashed-data KNN logic:

```text
collect smashed activations -> flatten -> optional PCA -> L2 normalize -> 1-NN with one known sample per class
```

## Setting

- Dataset: CIFAR-100
- Model: CIFAR-style ResNet-18
- Cuts:
  - C1: stem + layer1
  - C2: stem + layer1 + layer2
  - C3: stem + layer1 + layer2 + layer3
  - C4: stem + layer1 + layer2 + layer3 + layer4
- Attack: Sim-LIA smashed-data KNN
- Known reference data: 1 labeled sample per class
- Metrics: accuracy, macro-F1, weighted-F1, balanced accuracy

## Run

```bash
bash scripts/train_all_cuts.sh
bash scripts/attack_all_cuts.sh --batch-size 128 --n-attack 10000
```

Results are saved under:

```text
results/simlia/cifar100_resnet18_c*/
```
