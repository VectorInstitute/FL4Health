from typing import Any, Tuple

import torch


class BernoulliSample(torch.autograd.Function):
    @staticmethod
    def forward(bernoulli_probs: torch.Tensor) -> torch.Tensor:  # type: ignore
        return torch.bernoulli(input=bernoulli_probs)

    @staticmethod
    # inputs is a Tuple of all of the inputs passed to forward.
    # output is the output of the forward().
    def setup_context(ctx: Any, inputs: Tuple[torch.Tensor], output: torch.Tensor) -> None:
        assert len(inputs) == 1
        (bernoulli_probs,) = inputs
        ctx.save_for_backward(bernoulli_probs)

    # grad_output is supposed to be the gradient w.r.t. the output of the forward method.
    @staticmethod
    def backward(ctx: torch.Any, grad_output: torch.Tensor) -> torch.Tensor:  # type: ignore
        # ctx.saved_tensors is a tuple (of length 1 in this case). Hence the indexing here.
        bernoulli_probs = ctx.saved_tensors[0]
        return bernoulli_probs * grad_output


bernoulli_sample = BernoulliSample.apply
