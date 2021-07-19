from typing import Dict

import ignite.distributed as idist
from torch.utils.data import DataLoader

from config_schema import ConfigSchema
from data.weak_labels import WeakLabelManager

from .datasets import get_dataset


def get_dataflow(config: ConfigSchema, wlm: WeakLabelManager) -> Dict[str, DataLoader]:
    # - Get train/test datasets
    if idist.get_rank() > 0:
        # Ensure that only rank 0 download the dataset
        idist.barrier()

    dataset = get_dataset(config.dataset, config.data_path)
    train_split = wlm.convert_targets(dataset["train"])

    if idist.get_rank() == 0:
        # Ensure that only rank 0 download the dataset
        idist.barrier()

    # Setup data loader also adapted to distributed config: nccl, gloo, xla-tpu
    train_loader = idist.auto_dataloader(
        train_split,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
        drop_last=True,
    )
    val_loader = idist.auto_dataloader(
        dataset["val"],
        batch_size=2 * config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
    )
    test_loader = idist.auto_dataloader(
        dataset["test"],
        batch_size=2 * config.batch_size,
        num_workers=config.num_workers,
        shuffle=False,
    )
    return {"train": train_loader, "val": val_loader, "test": test_loader}
