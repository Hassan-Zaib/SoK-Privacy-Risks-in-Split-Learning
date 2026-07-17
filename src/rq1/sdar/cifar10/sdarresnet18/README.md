# SDAR ResNet-18 C1-C4

This folder is a lean source-only adaptation of the official SDAR implementation from `zhxchd/SDAR_SplitNN`.

Goal: keep SDAR faithful while changing only the controlled SAIA evaluation pieces:

- setting: vanilla Split Learning (`vsl`)
- dataset: CIFAR-10
- model: CIFAR-style ResNet-18
- cuts: `c1`, `c2`, `c3`, `c4`

Kept from official SDAR:

- TensorFlow/Keras implementation
- client/private data `D` and server auxiliary data `D'` split
- simulator `f_tilde`
- decoder `f_tilde^-1`
- simulator discriminator `d1`
- decoder/image discriminator `d2`
- conditional vanilla-SL use of labels
- Adam learning rates and SDAR hyperparameters
- 20,000 iterations for CIFAR-10
- batch size 128

ResNet-18 cuts:

- `c1`: stem = `conv1 + bn1 + relu`
- `c2`: stem + `layer1`
- `c3`: stem + `layer1 + layer2`
- `c4`: stem + `layer1 + layer2 + layer3`

Run one cut from the `scripts` directory:

```bash
cd scripts
python run_sdar.py vsl resnet18 cifar10 c1 1
```

Run all C1-C4:

```bash
cd scripts
bash run_sdar_resnet18_all.sh
```

Outputs are written under:

```text
scripts/sdar_results/vsl/cifar10/
```
