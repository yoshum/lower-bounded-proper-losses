import torch
from torch import Tensor
from torch.nn import Module


@torch.jit.script
def generalized_logit_squeezing(logits: Tensor, exponent: Tensor) -> Tensor:
    batch_size = logits.size(0)
    logits = logits - logits.mean(dim=1, keepdim=True)
    gls = (logits.abs() ** exponent).sum() / batch_size
    return gls


class GeneralizedLogitSqueezing(Module):
    def __init__(self, exponent: float):
        super().__init__()
        self.register_buffer("exponent", torch.tensor(exponent))

    def __call__(self, logits: Tensor) -> Tensor:
        return generalized_logit_squeezing(logits, self.exponent)
