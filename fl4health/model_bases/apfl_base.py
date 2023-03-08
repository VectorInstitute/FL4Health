import copy
from typing import Any, Callable, Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss


class APFLLoss:
    def __init__(self, losses: Dict[str, torch.Tensor]) -> None:
        self.losses = losses

    def __getitem__(self, item: str) -> torch.Tensor:
        return self.losses[item]

    def backward(self) -> None:
        self.losses["local"].backward()
        self.losses["global"].backward()


class APFLCriterion:
    def __init__(self, criterion: _Loss) -> None:
        self.local_criterion = criterion
        self.global_criterion = copy.deepcopy(criterion)
        self.personalized_criterion = copy.deepcopy(criterion)

    def __call__(self, model_outputs: Dict[str, torch.Tensor], ground_truth: torch.Tensor) -> APFLLoss:
        local_loss = self.local_criterion(model_outputs["local"], ground_truth)
        global_loss = self.global_criterion(model_outputs["global"], ground_truth)
        personalized_loss = self.personalized_criterion(model_outputs["personalized"], ground_truth)
        losses = {"local": local_loss, "global": global_loss, "personalized": personalized_loss}
        return APFLLoss(losses)


class APFLOptimizer:
    def __init__(self, model: Any, opt: Callable, **opt_kwargs: Any) -> None:
        self.model = model
        self.lr = 0.01
        self.local_opt = opt(self.model.local_model.parameters(), **opt_kwargs)
        self.global_opt = opt(self.model.global_model.parameters(), **opt_kwargs)

    def step(self) -> None:
        self.local_opt.step()
        self.global_opt.step()

        grad_alpha = 0.0
        for local_p, global_p in zip(self.model.local_model.parameters(), self.model.global_model.parameters()):
            dif = global_p - local_p
            grad = self.model.alpha * local_p.grad + (1 - self.model.alpha) * global_p.grad
            grad_alpha += dif.view(-1).T.dot(grad.view(-1))

        grad_alpha += 0.02 * self.model.alpha
        alpha_n = (self.model.alpha - self.lr * grad_alpha).detach().numpy()
        alpha_n = np.clip(alpha_n, 0, 1)
        self.model.alpha = alpha_n

    def zero_grad(self) -> None:
        self.local_opt.zero_grad()
        self.global_opt.zero_grad()


class APFLModule(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.local_model = model
        self.global_model = copy.deepcopy(model)
        self.alpha = 0.01

    def forward(self, input: torch.Tensor) -> Dict[str, torch.Tensor]:
        local_out = self.local_model(input)
        global_out = self.global_model(input)
        personalized_out = self.alpha * self.local_model(input) + (1 - self.alpha) * global_out
        return {"local": local_out, "global": global_out, "personalized": personalized_out}

    def layers_to_exchange(self) -> List[str]:
        layers_to_exchange = [layer for layer in self.state_dict().keys() if "global_model" in layer]
        return layers_to_exchange
