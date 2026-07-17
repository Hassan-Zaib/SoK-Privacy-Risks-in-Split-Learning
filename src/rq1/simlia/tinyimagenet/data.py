from __future__ import annotations

from pathlib import Path
import shutil

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def prepare_tiny_imagenet_val(root='data/tiny-imagenet-200'):
    root = Path(root)
    val_dir = root / 'val'
    images_dir = val_dir / 'images'
    ann_path = val_dir / 'val_annotations.txt'
    out_dir = val_dir / 'images_by_class'
    if out_dir.exists() or not ann_path.exists() or not images_dir.exists():
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    with ann_path.open('r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            filename, class_id = parts[0], parts[1]
            class_dir = out_dir / class_id
            class_dir.mkdir(parents=True, exist_ok=True)
            src = images_dir / filename
            dst = class_dir / filename
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)


def tinyimagenet_loaders(root='data/tiny-imagenet-200', batch_size=64, download=False, num_workers=2):
    root = Path(root)
    if download:
        # Torchvision does not provide a built-in Tiny-ImageNet downloader.
        # Put tiny-imagenet-200 under data/ or pass --data-root.
        pass
    prepare_tiny_imagenet_val(root)
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.ImageFolder(root / 'train', transform=transform)
    testset = datasets.ImageFolder(root / 'val' / 'images_by_class', transform=transform)
    trainloader = DataLoader(trainset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    testloader = DataLoader(testset, shuffle=False, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return trainset, testset, trainloader, testloader
