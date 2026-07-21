# MAERA ResNet-18 CIFAR-10

This folder implements a MAERA-style server-side data reconstruction attack for the SAIA experiments.

Faithful paper settings used here:

- Dataset: CIFAR-10
- Classes: 10
- Model: CIFAR-style ResNet-18
- Batch size: 128 adapted for our GPU/runtime. The MAERA paper uses 512.
- Main task training rounds/epochs: 100
- Shadow model and inversion model training rounds/epochs: 200
- Model learning rate: 0.001
- Discriminator learning rate: 0.001
- MAERA lambda: 1.0
- Minimum prior condition: 1% CIFAR-10 test samples + 1% private/client samples
- Reconstruction evaluation samples: 512

SAIA cut definitions:

- `c1`: client front = stem + layer1
- `c2`: client front = stem + layer1 + layer2
- `c3`: client front = stem + layer1 + layer2 + layer3
- `c4`: client front = stem + layer1 + layer2 + layer3 + layer4

U-shaped Split Learning layout:

- Client owns the front model `f_c1`.
- Server owns the middle model `f_s`.
- Client owns the classifier head `f_c2`.
- The server observes smashed activations `z = f_c1(x)`.

MAERA pipeline:

1. Train the normal U-shaped split model.
2. Store converged smashed activations from the victim/private client split.
3. Train a shadow front model `f_tilde_c1` and substitute head `f_h` with:
   - cross-entropy task loss on minimum-prior auxiliary CIFAR-10 data
   - WGAN-GP feature alignment between real smashed data and shadow smashed data
4. Train inversion decoder `f_tilde_c1^-1` with MSE on auxiliary data.
5. Reconstruct victim/test images from real smashed activations.

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
results/maera/cifar10_resnet18_c*/
results/logs/
checkpoints/
```
