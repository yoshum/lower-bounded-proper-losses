import torch
import torch.nn as nn
import torch.nn.functional as F

from config_schema import ConfigSchema
from data.weak_labels import WeakLabelManager

from .gls import GeneralizedLogitSqueezing


class WeakLabelLoss(nn.Module):
    def __init__(self, wlm: WeakLabelManager) -> None:
        super().__init__()
        self.num_classes = wlm.num_classes
        self.register_buffer("R", torch.from_numpy(wlm.R).to(dtype=torch.float))
        self.register_buffer("T", torch.from_numpy(wlm.T).to(dtype=torch.float))


class BackwardCorrectionLoss(WeakLabelLoss):
    def __init__(self, config: ConfigSchema, wlm: WeakLabelManager) -> None:
        super().__init__(wlm)
        self.base_loss = config.loss_func.base_loss
        self.kwargs = config.loss_func.kwargs

    def forward(self, scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.base_loss == "cross_entropy":
            log_p = F.log_softmax(scores, dim=1)
            Rlog_p = torch.mm(log_p, self.R)  # type: ignore
            return F.nll_loss(Rlog_p, targets)
        raise ValueError(f"unknown base loss type {self.base_loss}")


class BackwardCorrectionLogitSqueezingLoss(WeakLabelLoss):
    def __init__(self, config: ConfigSchema, wlm: WeakLabelManager) -> None:
        super().__init__(wlm)
        loss_cfg = config.loss_func
        self.base_loss = loss_cfg.base_loss

        assert all([key in ("coefficient", "exponent") for key in loss_cfg.kwargs])
        self.gls = GeneralizedLogitSqueezing(loss_cfg.kwargs.get("exponent", 2.0))
        self.register_buffer(
            "coefficient", torch.tensor(loss_cfg.kwargs["coefficient"]) / self.exponent
        )
        self.kwargs = loss_cfg.kwargs

    def forward(self, scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.base_loss == "cross_entropy":
            log_p = F.log_softmax(scores, dim=1)
            Rlog_p = torch.mm(log_p, self.R)  # type: ignore
            gls = self.gls(scores)
            return F.nll_loss(Rlog_p, targets) + self.coefficient * gls
        raise ValueError(f"unknown base loss type {self.base_loss}")


class BackwardCorrectionWithGA(WeakLabelLoss):
    def __init__(self, config: ConfigSchema, wlm: WeakLabelManager) -> None:
        super().__init__(wlm)
        self.base_loss = config.loss_func.base_loss

    def forward(self, scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.base_loss == "cross_entropy":
            log_p = F.log_softmax(scores, dim=1)
            R_t = self.R[:, targets].t()  # type: ignore
            Rlog_p = (log_p * R_t).mean(dim=0)
            return (-Rlog_p).abs().sum()
        raise ValueError(f"unknown base loss name {self.base_loss}")


def get_weak_label_loss(config: ConfigSchema, wlm: WeakLabelManager) -> nn.Module:
    loss_type = config.loss_func.loss_type
    if loss_type == "backward":
        return BackwardCorrectionLoss(config, wlm)
    elif loss_type == "backward_ga":
        return BackwardCorrectionWithGA(config, wlm)
    elif loss_type == "logit_squeezing":
        return BackwardCorrectionLogitSqueezingLoss(config, wlm)
    raise ValueError(f"unknown loss type {loss_type}")
