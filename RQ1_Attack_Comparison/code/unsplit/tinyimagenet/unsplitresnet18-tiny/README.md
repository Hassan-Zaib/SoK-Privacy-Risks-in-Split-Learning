# UnSplit ResNet-18 Tiny-ImageNet

This folder is copied from `unsplitresnet18` and keeps the same UnSplit-style attack/training settings while changing the dataset to Tiny-ImageNet.

Changed:
- Dataset: Tiny-ImageNet-200.
- Classes: 200.
- Result/checkpoint names: `tinyimagenet_resnet18_c1` to `tinyimagenet_resnet18_c4`.

Kept the same:
- CIFAR-style ResNet-18 stem: `3x3 conv`, stride 1, padding 1, no maxpool.
- Cuts: C1 stem, C2 stem+layer1, C3 stem+layer1+layer2, C4 stem+layer1+layer2+layer3.
- Batch size: 64.
- Epochs: 20.
- Train/test transform: `ToTensor()` only.
- Attack defaults: `n_attack=20`, `main_iters=1000`, `input_iters=100`, `model_iters=100`.
- `lambda_l2=0.0`, BatchNorm clone fix, and CIFAR-style min-max normalization before saving/metrics.

Expected data path:

```text
data/tiny-imagenet-200/
  train/
  val/
    images/
    val_annotations.txt
```

The loader automatically creates `val/images_by_class/` from `val_annotations.txt` for PyTorch `ImageFolder`.
