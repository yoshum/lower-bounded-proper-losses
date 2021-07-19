from datetime import datetime
from logging import Logger
from pathlib import Path
from typing import Dict

import ignite
import ignite.distributed as idist
import torch
from omegaconf import OmegaConf

from config_schema import ConfigSchema


def log_metrics(
    logger: Logger, epoch: int, elapsed: float, tag: str, metrics: Dict[str, float]
):
    logger.info(
        "Epoch {} - elapsed: {:.5f} - {} metrics: {}".format(
            epoch,
            elapsed,
            tag,
            ", ".join(["{}: {}".format(k, v) for k, v in metrics.items()]),
        )
    )


def log_basic_info(logger: Logger, config: ConfigSchema):
    logger.info("Experiment: {}".format(config.experiment_name))
    logger.info("- PyTorch version: {}".format(torch.__version__))
    logger.info("- Ignite version: {}".format(ignite.__version__))

    logger.info("\n")
    logger.info("Configuration:")
    for line in OmegaConf.to_yaml(config).split("\n"):
        logger.info("\t" + line)
    logger.info("\n")

    if idist.get_world_size() > 1:
        logger.info("\nDistributed setting:")
        logger.info("\tbackend: {}".format(idist.backend()))
        logger.info("\tworld size: {}".format(idist.get_world_size()))
        logger.info("\n")


def prepare_output_directory(config: ConfigSchema) -> None:
    formatted = datetime.now().strftime(config.output_path_format)

    output_path = Path(formatted)
    # force always to use a new directory to avoid overwriting existing ones
    output_path.mkdir(parents=True, exist_ok=False)
    config.output_path = output_path.as_posix()
