from __future__ import annotations

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def cifar100_loaders(root='data', batch_size=64, download=True, num_workers=2):
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.CIFAR100(root, download=download, train=True, transform=transform)
    testset = datasets.CIFAR100(root, download=download, train=False, transform=transform)
    trainloader = DataLoader(trainset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    testloader = DataLoader(testset, shuffle=False, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return trainset, testset, trainloader, testloader
