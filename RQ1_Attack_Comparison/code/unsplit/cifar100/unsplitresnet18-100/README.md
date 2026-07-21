# UnSplit ResNet-18 CIFAR-100

Official UnSplit-style code adapted for SAIA-SoK.

Kept from official UnSplit:
- `unsplit/attacks.py` has the same alternating model inversion / model stealing logic.
- Dummy image starts from 0.5.
- Adam optimizer with lr 0.001 and amsgrad=True for attack.
- Objective: MSE clone smashed activation + TV + L2.

Changed:
- Dataset: CIFAR-100.
- Architecture: CIFAR-style ResNet-18 with 100 output classes.
- Cuts: C1, C2, C3, C4.
- Training batch size: 64, matching official UnSplit.
- Metrics: MSE, PSNR, SSIM, LPIPS, ASR@SSIM>=0.90.

Train all cuts for 20 epochs:
```bash
bash scripts/train_all_cuts.sh --epochs 20
```

Attack all cuts:
```bash
bash scripts/attack_all_cuts.sh --n-attack 20 --main-iters 1000 --input-iters 100 --model-iters 100
```

Official-data setting: both CIFAR-100 train and test use ToTensor() only; no RandomCrop or RandomHorizontalFlip.

## Current ResNet-18 UnSplit Notes

This port keeps the official UnSplit alternating optimization loop, but uses CIFAR-100 ResNet-18 with these cuts:
- C1: stem only (`conv1 + bn1 + relu`)
- C2: stem + `layer1`
- C3: stem + `layer1 + layer2`
- C4: stem + `layer1 + layer2 + layer3`

For ResNet-18 BatchNorm, the clone is run in `eval()` mode and copies victim BatchNorm running statistics plus affine parameters. BN affine parameters are frozen during clone optimization. The default L2 image penalty is `0.0` because `lambda_l2=1.0` pushes CIFAR reconstructions toward near-zero/noise in this ResNet setup.


## CIFAR-100 Variant

This folder is copied from `unsplitresnet18` and changes only the dataset/output head to CIFAR-100: `datasets.CIFAR100`, `num_classes=100`, and `cifar100_resnet18_*` checkpoint/result names. Training transform remains `ToTensor()` only and batch size remains 64.
