from logging import Logger
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import ignite.distributed as idist
import torch
import torch.nn as nn
import torch.optim as optim
from ignite.contrib.engines import common
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events, create_supervised_evaluator
from ignite.handlers import Checkpoint, DiskSaver
from ignite.metrics import Accuracy, Loss, Metric
from ignite.utils import manual_seed, setup_logger
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from config_schema import ConfigSchema
from data.data_loader import get_dataflow
from data.weak_labels import WeakLabelManager, get_weak_label_manager
from handlers.lr_scheduler import get_lr_scheduler
from handlers.state_at_best_val import StateAtBestVal
from modules.losses import get_weak_label_loss
from modules.models import get_model
from utils.config import load_and_validate_config
from utils.logging import log_basic_info, log_metrics, prepare_output_directory


def run(config: ConfigSchema) -> None:
    spawn_kwargs = config.spawn_kwargs
    spawn_kwargs["nproc_per_node"] = config.nproc_per_node

    with idist.Parallel(backend=config.backend, **spawn_kwargs) as parallel:
        parallel.run(run_training, config)


def run_training(local_rank: int, config: ConfigSchema) -> Dict[str, float]:
    rank = idist.get_rank()
    if config.seed is not None:
        manual_seed(config.seed + rank)

    logger = setup_logger(name=config.experiment_name, distributed_rank=local_rank)

    log_basic_info(logger, config)

    if rank == 0:
        prepare_output_directory(config)
        logger.info("Output path: {}".format(config.output_path))

    weak_label_mgr = get_weak_label_manager(config)

    # Setup dataflow, model, optimizer, criterion
    data_loaders = get_dataflow(config, weak_label_mgr)
    train_loader = data_loaders["train"]
    config.num_iters_per_epoch = len(train_loader)

    model, optimizer, criterion = initialize(config, weak_label_mgr)

    metrics = get_metrics(criterion)
    trainer, evaluators = create_trainer_and_evaluators(
        model, optimizer, criterion, data_loaders, metrics, config, logger
    )

    if rank == 0:
        tb_logger = common.setup_tb_logging(
            config.output_path, trainer, optimizer, evaluators=evaluators
        )

    # Store 3 best models by validation accuracy:
    common.gen_save_best_models_by_val_score(
        save_handler=get_save_handler(config),
        evaluator=evaluators["val"],
        models={"model": model},
        metric_name="accuracy",
        n_saved=3,
        trainer=trainer,
        tag="test",
    )
    state_at_best_val = StateAtBestVal(
        score_function=lambda: evaluators["val"].state.metrics["accuracy"],
        state_function=lambda: dict(
            {"val_" + key: val for key, val in evaluators["val"].state.metrics.items()},
            **{
                "test_" + key: val
                for key, val in evaluators["test"].state.metrics.items()
            },
            epoch=trainer.state.epoch
        ),
    )
    trainer.add_event_handler(Events.EPOCH_COMPLETED, state_at_best_val)

    try:
        trainer.run(train_loader, max_epochs=config.num_epochs)
    except Exception:
        import traceback

        print(traceback.format_exc())
    else:
        assert state_at_best_val.best_state is not None
        tb_logger.writer.add_hparams(  # type: ignore
            get_hparams(config),
            {"hparam/" + key: val for key, val in state_at_best_val.best_state.items()},
        )
    finally:
        if rank == 0:
            tb_logger.close()  # type: ignore

    return evaluators["val"].state.metrics


def initialize(
    config: ConfigSchema, wlm: WeakLabelManager
) -> Tuple[nn.Module, Optimizer, nn.Module]:
    model = get_model(config.model)
    # Adapt model for distributed settings if configured
    model = idist.auto_model(model)

    to_decay, not_to_deacy = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        elif len(param.shape) == 1 or name.endswith("bias"):
            not_to_deacy.append(param)
        else:
            to_decay.append(param)
    optimizer = optim.SGD(
        [
            {"params": to_decay, "weight_decay": config.weight_decay},
            {"params": not_to_deacy, "weight_decay": 0.0},
        ],
        lr=config.learning_rate,
        momentum=config.momentum,
        nesterov=True,
    )
    optimizer = idist.auto_optim(optimizer)
    criterion = get_weak_label_loss(config, wlm).to(idist.device())

    return model, optimizer, criterion


def create_trainer_and_evaluators(
    model: nn.Module,
    optimizer: Optimizer,
    criterion: nn.Module,
    data_loaders: Dict[str, DataLoader],
    metrics: Dict[str, Metric],
    config: ConfigSchema,
    logger: Logger,
) -> Tuple[Engine, Dict[str, Engine]]:
    trainer = get_trainer(model, criterion, optimizer)
    trainer.logger = logger

    evaluators = get_evaluators(model, metrics)
    setup_evaluation(trainer, evaluators, data_loaders, logger)

    lr_scheduler = get_lr_scheduler(config, optimizer, trainer, evaluators["val"])

    to_save = {
        "trainer": trainer,
        "model": model,
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler,
    }

    common.setup_common_training_handlers(
        trainer=trainer,
        to_save=to_save,
        save_every_iters=config.checkpoint_every,
        save_handler=get_save_handler(config),
        with_pbars=False,
        train_sampler=data_loaders["train"].sampler,
    )
    trainer.add_event_handler(Events.EPOCH_STARTED, lr_scheduler)
    ProgressBar(persist=False).attach(
        trainer,
        metric_names="all",
        event_name=Events.ITERATION_COMPLETED(every=config.log_every_iters),
    )

    resume_from = config.resume_from
    if resume_from is not None:
        checkpoint_fp = Path(resume_from)
        assert checkpoint_fp.exists(), "Checkpoint '{}' is not found".format(
            checkpoint_fp.as_posix()
        )
        logger.info("Resume from a checkpoint: {}".format(checkpoint_fp.as_posix()))
        checkpoint = torch.load(checkpoint_fp.as_posix(), map_location="cpu")
        Checkpoint.load_objects(to_load=to_save, checkpoint=checkpoint)

    return trainer, evaluators


def setup_evaluation(
    trainer: Engine,
    evaluators: Dict[str, Engine],
    data_loaders: Dict[str, DataLoader],
    logger: Logger,
) -> None:
    # We define two evaluators as they wont have exactly similar roles:
    # - `evaluator` will save the best model based on validation score
    def _evaluation(engine: Engine) -> None:
        epoch = trainer.state.epoch
        for split in ["train", "val", "test"]:
            state = evaluators[split].run(data_loaders[split])
            log_metrics(logger, epoch, state.times["COMPLETED"], split, state.metrics)

    trainer.add_event_handler(
        Events.EPOCH_COMPLETED(every=config.validate_every) | Events.COMPLETED,
        _evaluation,
    )
    return


def get_trainer(model: nn.Module, criterion: Callable, optimizer: Optimizer) -> Engine:
    device = idist.device()

    def train_step(engine: Engine, batch: Tuple[torch.Tensor, torch.Tensor]) -> float:

        x, y = batch[0], batch[1]

        if x.device != device:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

        model.train()
        # Supervised part
        y_pred = model(x)
        loss = criterion(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    trainer = Engine(train_step)
    return trainer


def get_evaluators(model: nn.Module, metrics: Dict[str, Metric]) -> Dict[str, Engine]:
    test_evaluator = create_supervised_evaluator(
        model,
        metrics={"accuracy": metrics["accuracy"]},
        device=idist.device(),
        non_blocking=True,
    )
    val_evaluator = create_supervised_evaluator(
        model,
        metrics={"accuracy": metrics["accuracy"]},
        device=idist.device(),
        non_blocking=True,
    )
    train_evaluator = create_supervised_evaluator(
        model,
        metrics={"loss": metrics["loss"]},
        device=idist.device(),
        non_blocking=True,
    )
    return {"train": train_evaluator, "val": val_evaluator, "test": test_evaluator}


def get_metrics(loss: Callable) -> Dict[str, Metric]:
    return {"accuracy": Accuracy(), "loss": Loss(loss)}


def get_save_handler(config) -> DiskSaver:
    return DiskSaver(config["output_path"], require_empty=False)


def get_hparams(config: ConfigSchema) -> Dict[str, Any]:
    hparams = {
        "model": config.model,
        "learning_rate": config.learning_rate,
        "momentum": config.momentum,
        "weight_decay": config.weight_decay,
        "batch_size": config.batch_size,
        "loss_type": config.loss_func.loss_type,
        "coeff": config.loss_func.kwargs.get("coeff", 0.0),
        "exponent": config.loss_func.kwargs.get("exponent", 2.0),
        "patience": config.patience,
    }
    hparams.update(config.loss_func.kwargs)
    return hparams


if __name__ == "__main__":
    config = load_and_validate_config(ConfigSchema)
    run(config)  # type: ignore
