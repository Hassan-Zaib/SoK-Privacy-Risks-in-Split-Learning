# RQ3 ConvNeXt-Tiny Split Learning on Tiny ImageNet

This folder runs the RQ3 experiment: original ConvNeXt-Tiny style backbone on native Tiny ImageNet 64x64 with post-cut activation quantization.

## Architecture

- Backbone: ConvNeXt-Tiny depths `[3, 3, 9, 3]`, dims `[96, 192, 384, 768]`.
- Dataset: Tiny ImageNet, 200 classes, native `64x64`.
- Training: 2-party sequential Split Learning.
- Main defenses: `none`, `q4`, `q2`, `noise0.01`, `noise0.05`.
- Fixed probe: created once in `fixed_probe/` and reused for all current and future defenses.

The public cut names are stage-level only:

| Cut | Location | Expected Smashed Shape |
|---|---|---|
| C1 | after stage 1 | `[B, 96, 16, 16]` |
| C2 | after stage 2 | `[B, 192, 8, 8]` |
| C3 | after stage 3 | `[B, 384, 4, 4]` |
| C4 | after stage 4 | `[B, 768, 2, 2]` |

The ConvNeXt stem is internal and is not reported as a main cut.

## Run on server

From your local machine:

```bash
```

On the server:

```bash
mkdir -p data
ln -sfn ~/saia_sok/datasets/tiny-imagenet-200 data/tiny-imagenet-200
```

Start the full sequential run:

```bash
```

Attach:

```bash
tmux attach -t rq3_convnext_quant_seq
```

The default full recipe is:

```text
300 epochs
batch size 256
AdamW, learning rate 0.004
20 warmup epochs followed by cosine decay
Mixup/CutMix, RandAugment, Random Erasing
EMA decay 0.9999
20,000 full SDAR joint-training steps, batch size 128
```

The runner executes in this order:

1. fixed probe + activation shape smoke checks
2. small smoke training/attack on `none C1`
3. `none`: C1, C2, C3, C4
4. `q2`: C1, C2, C3, C4
5. `q4`: C1, C2, C3, C4
6. `noise0.01`: C1, C2, C3, C4
6. `noise0.05`: C1, C2, C3, C4

Completed victim and attack summaries are detected and skipped, so the same command can resume an interrupted run.

After each defense group, it prints the table and saves a fixed-probe sheet:

```text
results/probe_sheets/rq3_convnext_tiny_<defense>_fixed_probe_orig_c1_c4.png
```

## Notes

The privacy run follows the official SDAR protocol independently of the
300-epoch utility run. It initializes victim `f,g`, simulator `e`, feature
discriminator, conditional decoder, and image discriminator together, then
updates them online for 20,000 steps. For defended settings, both the shared
server and attacker observe post-cut `z_tilde`, not clean `z`.

The standalone validation accuracy and the SDAR online victim accuracy are
reported separately. They are not the same trained model and must not be
presented as one measurement.
