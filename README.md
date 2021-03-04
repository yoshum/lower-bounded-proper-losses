# Lower-bounded proper losses for weakly supervised classification

This directory contains codes for experiments in the following paper: **to appear**

## Usage

We recommend using Poetry to install the dependencies and run experiments.
Alternatively, you can manually install the packages listed in `pyproject.toml`.
Here, we only describe the path with Poetry.

> **NOTE:**  All the commands below are supposed to be executed in this directory

### Requirements

- Python 3.8
- Poetry 1.1.4

### Setup

Run `poetry install` to install all the dependencies

### Run an experiment

Execute `poetry run python -m main config=/path/to/config/file.yml`.

You can override settings in a config file by giving additional runtime arguments as follows:

- `poetry run python -m main config=config/cifar10-wrn_28_2/logit_squeezing.yml learning_rate=0.1`
- `poetry run python -m main config=config/cifar10-wrn_28_2/logit_squeezing.yml loss_func.kwargs.exponent=1.1`

See the `config` directory for example config files.
Other configurable items are listed in `config_schema.py` as the data class `ConfigSchema`.
