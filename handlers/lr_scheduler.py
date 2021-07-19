from ignite.contrib.handlers import (
    CosineAnnealingScheduler,
    create_lr_scheduler_with_warmup,
)
from ignite.engine import Engine
from torch.optim import Optimizer

from config_schema import ConfigSchema
from handlers.lr_reduction_early_stopping import LRReductionEarlyStopping


def get_lr_scheduler(
    config: ConfigSchema, optimizer: Optimizer, trainer: Engine, evaluator: Engine
):
    if config.num_warmup_epochs:
        length = config.num_epochs - config.num_warmup_epochs
    else:
        length = config.num_epochs

    if config.lr_scheduler == "cosine":
        lr_scheduler = CosineAnnealingScheduler(
            optimizer,
            "lr",
            config.learning_rate,
            0.001 * config.learning_rate,
            cycle_size=length + 1,
        )
        if config.num_warmup_epochs:
            lr_scheduler = create_lr_scheduler_with_warmup(
                lr_scheduler, 0.0, config.num_warmup_epochs
            )
    elif config.lr_scheduler == "reduce_at_plateau":
        lr_scheduler = LRReductionEarlyStopping(
            optimizer,
            trainer=trainer,
            reduction_rate=0.1,
            num_reduction=2,
            patience=config.patience,
            score_function=lambda _: evaluator.state.metrics["accuracy"],
            num_warmup_epochs=config.num_warmup_epochs,
            warmup_start_value=0.001 * config.learning_rate,
        )
    else:
        raise ValueError(f"unknown lr scheduler {config.lr_scheduler}")

    return lr_scheduler
