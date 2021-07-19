import logging
from copy import copy
from typing import Callable, List, Optional, OrderedDict

from ignite.engine import Engine
from torch.optim import Optimizer


class LRReductionEarlyStopping:
    def __init__(
        self,
        optimizer: Optimizer,
        reduction_rate: float,
        num_reduction: int,
        patience: int,
        score_function: Callable,
        trainer: Engine,
        num_warmup_epochs: int = 0,
        warmup_start_value: float = 0.0,
        min_delta: float = 0.0,
        cumulative_delta: bool = False,
    ):
        self.optimizer = optimizer
        self.optimizer_param_groups = optimizer.param_groups
        self.reduction_rate = reduction_rate
        self.num_reduction = num_reduction
        self.patience = patience
        self.score_function = score_function
        self.trainer = trainer
        self.num_warmup_epochs = num_warmup_epochs
        self.warmup_start_value = warmup_start_value
        self.initial_lrs = [param_group["lr"] for param_group in optimizer.param_groups]
        self.min_delta = min_delta
        self.cumulative_delta = cumulative_delta

        self.best_score: Optional[float] = None
        self.patience_counter = 0
        self.reduction_counter = 0

        self.logger = logging.getLogger(__name__ + "." + self.__class__.__name__)

        self._state_attrs = ["best_score", "patience_counter", "reduction_counter"]

    def reduce_lr(self) -> None:
        for param_group in self.optimizer_param_groups:
            param_group["lr"] *= self.reduction_rate

    def set_lr(self, lrs: List[float]) -> None:
        for param_group, lr in zip(self.optimizer_param_groups, lrs):
            param_group["lr"] = lr

    def __call__(self, engine: Engine) -> None:
        epoch = self.trainer.state.epoch
        num_warmup_epochs = self.num_warmup_epochs
        if num_warmup_epochs and epoch <= num_warmup_epochs:
            lrs = [
                (initial_lr - self.warmup_start_value) * epoch / num_warmup_epochs
                + self.warmup_start_value
                for initial_lr in self.initial_lrs
            ]
            self.set_lr(lrs)
        elif self.trigger(engine):
            if self.reduction_counter < self.num_reduction:
                self.logger.info("To reduce the learning rate")
                self.reduce_lr()
                self.reduction_counter += 1
                self.patience_counter = 0
            else:
                self.logger.info("To terminate training")
                self.trainer.terminate()

    def trigger(self, engine: Engine) -> bool:
        score = self.score_function(engine)

        if self.best_score is None:
            self.best_score = score
        elif score <= self.best_score + self.min_delta:
            if not self.cumulative_delta and score > self.best_score:
                self.best_score = score
            self.patience_counter += 1
            self.logger.debug(
                "%i / %i (patience), %i / %i (reduction)"
                % (
                    self.patience_counter,
                    self.patience,
                    self.reduction_counter,
                    self.num_reduction,
                )
            )
            if self.patience_counter >= self.patience:
                return True
        else:
            self.best_score = score
            self.patience_counter = 0

        return False

    def state_dict(self):
        state = OrderedDict()
        for name in self._state_attrs:
            val = getattr(self, name)
            if hasattr(val, "state_dict"):
                val = val.state_dict()
            state[name] = copy(val)
        return state

    def load_state_dict(self, state_dict):
        for name in self._state_attrs:
            obj = getattr(self, name)
            if hasattr(obj, "load_state_dict"):
                obj.load_state_dict(state_dict[name])
            else:
                setattr(self, name, state_dict[name])
