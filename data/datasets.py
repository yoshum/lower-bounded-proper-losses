import os
from typing import Dict

from torch.utils.data import Dataset, random_split
from torchvision import datasets
from torchvision.transforms import (
    Compose,
    Normalize,
    Pad,
    RandomCrop,
    RandomHorizontalFlip,
    ToTensor,
)

train_transform = Compose(
    [
        Pad(4),
        RandomCrop(32, fill=128),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

test_transform = Compose(
    [
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)


def get_dataset(name: str, path: str) -> Dict[str, Dataset]:
    if not os.path.exists(path):
        os.makedirs(path)

    if name == "cifar10":
        train_org = datasets.CIFAR10(
            root=path, train=True, download=True, transform=train_transform
        )
        test_ds = datasets.CIFAR10(
            root=path, train=False, download=False, transform=test_transform
        )
    elif name == "mnist":
        train_org = datasets.MNIST(
            root=path, train=True, download=True, transform=ToTensor()
        )
        test_ds = datasets.MNIST(
            root=path, train=False, download=False, transform=ToTensor()
        )
    else:
        raise ValueError(f"unknown dataset type {name}")

    train_ds, val_ds = random_split(
        train_org,
        [len(train_org) - int(0.1 * len(train_org)), int(0.1 * len(train_org))],
    )
    return {"train": train_ds, "val": val_ds, "test": test_ds}
