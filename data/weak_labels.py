from typing import Union

import numpy as np
from torch.utils.data import Subset
from torchvision.datasets import VisionDataset

from config_schema import ConfigSchema

DatasetTypes = Union[VisionDataset, Subset]


class WeakLabelManager:
    def __init__(self, num_classes: int, **kwargs):
        self.num_classes = num_classes

    def convert_targets(self, dataset: DatasetTypes) -> DatasetTypes:
        pass

    @property
    def T(self) -> np.ndarray:
        pass

    @property
    def R(self) -> np.ndarray:
        pass


def get_weak_label_manager(config: ConfigSchema) -> WeakLabelManager:
    if config.weak_label_type == "complementary_labels":
        return ComplementaryLabelManager(config.num_classes)
    raise ValueError(f"unknown weak-label type {config.weak_label_type}")


class ComplementaryLabelManager(WeakLabelManager):
    @property
    def T(self) -> np.ndarray:
        num_classes = self.num_classes
        T = np.ones((num_classes, num_classes)) / (num_classes - 1)
        for i in range(num_classes):
            T[i, i] = 0

        return T

    @property
    def R(self) -> np.ndarray:
        return np.linalg.inv(self.T)

    def random_comp_labels(self, labels: np.ndarray, num_classes: int) -> np.ndarray:
        K1 = num_classes - 1
        comp_labels = np.random.randint(
            K1, size=len(labels)
        )  # random numbers from 0 to #classes-2
        comp_labels[comp_labels >= labels] += 1
        return comp_labels

    def convert_targets(self, dataset: DatasetTypes) -> DatasetTypes:
        labels = get_targets(dataset)
        comp_labels = self.random_comp_labels(labels, self.num_classes)
        replace_targets(dataset, comp_labels)
        return dataset


def get_targets(dataset: DatasetTypes) -> np.ndarray:
    targets = [dataset[i][1] for i in range(len(dataset))]
    return np.array(targets)


def replace_targets(dataset: DatasetTypes, targets: np.ndarray) -> None:
    assert len(dataset) == len(targets)
    if isinstance(dataset, Subset):
        assert isinstance(dataset.dataset, VisionDataset)
        for idx, target in zip(dataset.indices, targets):
            dataset.dataset.targets[idx] = target
    else:
        assert isinstance(dataset, VisionDataset)
        dataset.targets = targets
