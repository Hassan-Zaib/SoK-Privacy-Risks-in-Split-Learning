# Dataset Setup

This artifact uses CIFAR-10, CIFAR-100, and Tiny ImageNet.

## CIFAR-10 and CIFAR-100

The CIFAR-10 and CIFAR-100 experiments use the standard torchvision datasets. In the provided scripts, these datasets are downloaded automatically when supported by the local data loader.

Expected local layout after download:

```text
data/
  cifar-10-batches-py/
  cifar-100-python/
```

## Tiny ImageNet

Tiny ImageNet must be downloaded manually and placed under the experiment directory or linked from a shared dataset directory.

Expected layout:

```text
data/tiny-imagenet-200/
  train/
  val/
  wnids.txt
  words.txt
```

On the server used for the paper experiments, Tiny ImageNet was shared through a symbolic link when multiple experiment folders used the same dataset.

Example:

```bash
mkdir -p data
ln -s /path/to/tiny-imagenet-200 data/tiny-imagenet-200
```

Raw datasets are not included in this repository.
