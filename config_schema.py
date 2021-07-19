from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from omegaconf import MISSING


@dataclass
class LossFuncConfig:
    # loss_type = "backward", "backward_ga", or "logit_squeezing"
    loss_type: str = MISSING

    # set "exponent" and "coefficient" when loss_type="logit_squeezing"
    kwargs: Dict[str, Any] = field(default_factory=dict)

    base_loss: str = "cross_entropy"


@dataclass
class BaseSchema:
    seed: Optional[int] = None
    data_path: str = "${env:HOME}/.cache/torchvision/data"
    output_path: str = MISSING  # set automatically by a script
    batch_size: int = 256
    momentum: float = 0.9
    weight_decay: float = 1e-4
    num_workers: int = 4
    num_epochs: int = 1000
    learning_rate: float = 0.01
    num_warmup_epochs: int = 4
    validate_every: int = 1
    patience: int = 10
    checkpoint_every: int = 1000
    backend: Optional[str] = None
    resume_from: Optional[str] = None
    log_every_iters: int = 1
    nproc_per_node: Optional[int] = None
    stop_iteration: Optional[int] = None
    with_trains: bool = False
    lr_scheduler: str = "reduce_at_plateau"
    spawn_kwargs: Dict[str, Any] = field(default_factory=dict)

    num_iters_per_epoch: int = MISSING


@dataclass
class ConfigSchema(BaseSchema):
    experiment_name: str = (
        "${weak_label_type}-${loss_func.loss_type}-${dataset}-${model}"
    )
    output_path_format: str = (
        "/workspace/results/"
        "${weak_label_type}/${dataset}-${model}/${loss_func.loss_type}/%Y%m%d-%H%M%S"
    )

    weak_label_type: str = "complementary_labels"
    num_classes: int = 10

    # dataset = "cifar10" or "mnist"
    dataset: str = "cifar10"

    # model = "linear", "mlp500", "resnet20", or "wrn_28_2"
    model: str = "resnet18"
    loss_func: LossFuncConfig = LossFuncConfig()
