from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision import datasets, transforms


def _prepare_tiny_val(root: Path):
    val_dir = root / "val"
    image_dir = val_dir / "images"
    class_dir = val_dir / "images_by_class"
    annot = val_dir / "val_annotations.txt"
    if class_dir.exists():
        return class_dir
    class_dir.mkdir(parents=True, exist_ok=True)
    mapping = {}
    with annot.open("r", encoding="utf-8") as f:
        for line in f:
            image_name, class_name = line.split("\t")[:2]
            mapping[image_name] = class_name
    for image_name, class_name in mapping.items():
        target = class_dir / class_name
        target.mkdir(exist_ok=True)
        src = image_dir / image_name
        dst = target / image_name
        if src.exists() and not dst.exists():
            dst.symlink_to(src)
    return class_dir


def tinyimagenet_private_aux_loaders(
    root: str = "data/tiny-imagenet-200",
    batch_size: int = 128,
    test_aux_fraction: float = 0.50,
    private_aux_fraction: float = 0.01,
    seed: int = 123,
    num_workers: int = 4,
):
    root_path = Path(root)
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.ImageFolder(root_path / "train", transform=transform)
    valset = datasets.ImageFolder(_prepare_tiny_val(root_path), transform=transform)

    train_perm = torch.randperm(len(trainset), generator=torch.Generator().manual_seed(seed)).tolist()
    val_perm = torch.randperm(len(valset), generator=torch.Generator().manual_seed(seed + 1)).tolist()

    private_aux_size = max(1, int(len(trainset) * private_aux_fraction))
    test_aux_size = max(1, int(len(valset) * test_aux_fraction))

    private_aux_indices = train_perm[:private_aux_size]
    private_indices = train_perm[private_aux_size:]
    test_aux_indices = val_perm[:test_aux_size]

    private_set = Subset(trainset, private_indices)
    aux_set = ConcatDataset(
        [
            Subset(valset, test_aux_indices),
            Subset(trainset, private_aux_indices),
        ]
    )

    private_loader = DataLoader(
        private_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    aux_loader = DataLoader(
        aux_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        valset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    prior_info = {
        "test_aux_fraction": test_aux_fraction,
        "private_aux_fraction": private_aux_fraction,
        "test_aux_size": test_aux_size,
        "private_aux_size": private_aux_size,
        "private_train_size": len(private_set),
        "aux_size": len(aux_set),
    }
    return trainset, valset, private_loader, aux_loader, test_loader, prior_info


def tinyimagenet_attack_subset(root: str = "data/tiny-imagenet-200", n_attack: int = 512, seed: int = 123):
    root_path = Path(root)
    transform = transforms.Compose([transforms.ToTensor()])
    valset = datasets.ImageFolder(_prepare_tiny_val(root_path), transform=transform)
    attack_indices = torch.randperm(
        len(valset),
        generator=torch.Generator().manual_seed(seed + 2),
    ).tolist()[: min(n_attack, len(valset))]
    return Subset(valset, attack_indices)
