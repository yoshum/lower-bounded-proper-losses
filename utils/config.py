from typing import Type

from omegaconf import DictConfig, OmegaConf


def load_and_validate_config(config_schema: Type) -> DictConfig:
    config_cli = OmegaConf.from_cli()
    if config_cli.config:
        config_yml = OmegaConf.load(config_cli.config)
        del config_cli["config"]
    else:
        config_yml = OmegaConf.create({})

    return OmegaConf.merge(config_schema, config_yml, config_cli)  # type: ignore
