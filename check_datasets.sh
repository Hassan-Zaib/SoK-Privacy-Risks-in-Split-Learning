#!/usr/bin/env bash
set -euo pipefail

TINY_ROOT="${TINY_IMAGENET_ROOT:-${DATA_ROOT:-}}"
if [ -z "$TINY_ROOT" ]; then
  for d in \
    "$PWD/data/tiny-imagenet-200" \
    "$PWD/RQ3_Architecture_Defenses/data/tiny-imagenet-200" \
    "$HOME/saia_sok/datasets/tiny-imagenet-200"
  do
    if [ -d "$d" ]; then
      TINY_ROOT="$d"
      break
    fi
  done
fi

echo "CIFAR-10: torchvision can download automatically if network is available"
echo "CIFAR-100: torchvision can download automatically if network is available"

if [ -z "$TINY_ROOT" ] || [ ! -d "$TINY_ROOT" ]; then
  echo "[FAIL] Tiny ImageNet not found"
  echo "Set TINY_IMAGENET_ROOT=/path/to/tiny-imagenet-200"
  exit 1
fi

test -f "$TINY_ROOT/wnids.txt"
test -f "$TINY_ROOT/val/val_annotations.txt"
test -d "$TINY_ROOT/train"
test -d "$TINY_ROOT/val/images"

train_count=$(find "$TINY_ROOT/train" -type f \( -name "*.JPEG" -o -name "*.jpg" -o -name "*.png" \) | wc -l)
val_count=$(find "$TINY_ROOT/val/images" -type f \( -name "*.JPEG" -o -name "*.jpg" -o -name "*.png" \) | wc -l)

echo "Tiny ImageNet root: $TINY_ROOT"
echo "Tiny ImageNet training images: $train_count"
echo "Tiny ImageNet validation images: $val_count"
echo "[PASS] Dataset validation"
