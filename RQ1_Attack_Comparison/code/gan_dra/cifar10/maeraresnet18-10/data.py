from __future__ import annotations

import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision import datasets, transforms


def cifar10_private_aux_loaders(
    root: str = "data",
    batch_size: int = 128,
    test_aux_fraction: float = 0.01,
    private_aux_fraction: float = 0.01,
    seed: int = 123,
    download: bool = True,
    num_workers: int = 4,
):
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.CIFAR10(root, train=True, download=download, transform=transform)
    testset = datasets.CIFAR10(root, train=False, download=download, transform=transform)

    generator = torch.Generator().manual_seed(seed)
    train_perm = torch.randperm(len(trainset), generator=generator).tolist()
    test_perm = torch.randperm(len(testset), generator=torch.Generator().manual_seed(seed + 1)).tolist()

    private_aux_size = max(1, int(len(trainset) * private_aux_fraction))
    test_aux_size = max(1, int(len(testset) * test_aux_fraction))

    private_aux_indices = train_perm[:private_aux_size]
    private_indices = train_perm[private_aux_size:]
    test_aux_indices = test_perm[:test_aux_size]

    private_set = Subset(trainset, private_indices)
    aux_set = ConcatDataset(
        [
            Subset(testset, test_aux_indices),
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
        testset,
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
    return trainset, testset, private_loader, aux_loader, test_loader, prior_info


def cifar10_attack_subset(root: str = "data", n_attack: int = 128, download: bool = True):
    transform = transforms.Compose([transforms.ToTensor()])
    testset = datasets.CIFAR10(root, train=False, download=download, transform=transform)
    return Subset(testset, list(range(min(n_attack, len(testset)))))
