# MAERA ResNet-18 Tiny ImageNet

This folder implements a MAERA-style server-side data reconstruction attack for Tiny ImageNet.

Settings used for Tiny ImageNet:

- Dataset: Tiny ImageNet
- Classes: 200
- Input size: 64x64
- Model: ResNet-18
- Main task epochs: 200
- Shadow model epochs: 200
- Inversion decoder epochs: 200
- Model learning rate: 0.001
- Discriminator learning rate: 0.001
- MAERA lambda: 1.0
- Tiny ImageNet prior condition: 50% validation/test samples + 1% private/client samples

Runtime adaptation:

- Batch size: 128. The MAERA paper uses 512.
- Reconstruction evaluation samples: 512.

SAIA cut definitions:

- `c1`: client front = stem + layer1
- `c2`: client front = stem + layer1 + layer2
- `c3`: client front = stem + layer1 + layer2 + layer3
- `c4`: client front = stem + layer1 + layer2 + layer3 + layer4

Feature shapes for 64x64 Tiny ImageNet:

- `c1`: 64 channels, 64x64
- `c2`: 128 channels, 32x32
- `c3`: 256 channels, 16x16
- `c4`: 512 channels, 8x8

U-shaped Split Learning layout:

- Client owns the front model `f_c1`.
- Server owns the middle model `f_s`.
- Client owns the classifier head `f_c2`.
- The server observes smashed activations `z = f_c1(x)`.

MAERA pipeline:

1. Train the normal U-shaped split model.
2. Store converged smashed activations from the victim/private client split.
3. Train a shadow front model `f_tilde_c1` and substitute head `f_h` with:
   - cross-entropy task loss on Tiny ImageNet auxiliary data
   - WGAN-GP feature alignment between real smashed data and shadow smashed data
4. Train inversion decoder `f_tilde_c1^-1` with MSE on auxiliary data.
5. Reconstruct victim/validation images from real smashed activations.

Before running, place or symlink Tiny ImageNet at:

```text
data/tiny-imagenet-200
```

Run a smoke test:

```bash
bash scripts/run_maera_smoke.sh
```

Run one cut:

```bash
python run_maera.py --cut c1
```

Run all cuts:

```bash
bash scripts/run_maera_all_cuts.sh
```

Outputs:

```text
results/maera/tinyimagenet_resnet18_c*/
results/logs/
checkpoints/
```
