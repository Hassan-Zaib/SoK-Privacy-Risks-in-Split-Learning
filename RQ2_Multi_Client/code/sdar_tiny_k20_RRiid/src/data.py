from pathlib import Path
import os
import shutil

import numpy as np
import tensorflow as tf
from PIL import Image


def _resolve_root(root=None):
    if root is not None:
        return Path(root)
    env_root = os.environ.get("TINY_IMAGENET_ROOT")
    if env_root:
        return Path(env_root)
    return Path("data/tiny-imagenet-200")


def prepare_tiny_imagenet_val(root):
    root = Path(root)
    val_dir = root / "val"
    images_dir = val_dir / "images"
    ann_path = val_dir / "val_annotations.txt"
    out_dir = val_dir / "images_by_class"

    if out_dir.exists():
        return
    if not ann_path.exists() or not images_dir.exists():
        raise FileNotFoundError("Tiny ImageNet validation files are missing.")

    out_dir.mkdir(parents=True, exist_ok=True)
    with ann_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            filename, class_id = parts[0], parts[1]
            class_dir = out_dir / class_id
            class_dir.mkdir(parents=True, exist_ok=True)
            src = images_dir / filename
            dst = class_dir / filename
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)


def _class_to_index(root):
    train_dir = Path(root) / "train"
    classes = sorted(p.name for p in train_dir.iterdir() if p.is_dir())
    if not classes:
        raise RuntimeError(f"No Tiny ImageNet classes found under {train_dir}")
    return {name: idx for idx, name in enumerate(classes)}


def _read_image(path):
    img = Image.open(path).convert("RGB")
    if img.size != (64, 64):
        img = img.resize((64, 64), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def _load_split(root, class_to_idx, split):
    xs = []
    ys = []

    if split == "train":
        for class_id, label in class_to_idx.items():
            img_dir = Path(root) / "train" / class_id / "images"
            for path in sorted(img_dir.glob("*")):
                if path.suffix.lower() in {".jpeg", ".jpg", ".png"}:
                    xs.append(_read_image(path))
                    ys.append(label)
    else:
        prepare_tiny_imagenet_val(root)
        for class_id, label in class_to_idx.items():
            img_dir = Path(root) / "val" / "images_by_class" / class_id
            for path in sorted(img_dir.glob("*")):
                if path.suffix.lower() in {".jpeg", ".jpg", ".png"}:
                    xs.append(_read_image(path))
                    ys.append(label)

    if not xs:
        raise RuntimeError(f"No Tiny ImageNet images found for split {split}")

    return np.stack(xs, axis=0), np.asarray(ys, dtype=np.int64).reshape(-1, 1)


def load_tiny_imagenet(root=None):
    root = _resolve_root(root)
    if not root.exists():
        raise FileNotFoundError(
            f"Tiny ImageNet not found at {root}. Set TINY_IMAGENET_ROOT or create data/tiny-imagenet-200."
        )

    class_to_idx = _class_to_index(root)
    train = _load_split(root, class_to_idx, "train")
    val = _load_split(root, class_to_idx, "val")
    return train, val


def make_dataset(x, y, transform=None, seed=42):
    x = x.astype(np.float32)
    y = y.astype(np.int64)

    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if transform is not None:
        ds = ds.map(lambda a, b: (transform(a), b), num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.shuffle(min(len(x), 10000), seed=seed, reshuffle_each_iteration=True)
    return ds
