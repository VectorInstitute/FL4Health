from typing import Any, Tuple

import torch


class BernoulliSample(torch.autograd.Function):
    """
    Bernoulli sampling function that allows for gradient computation.

    Bernoulli sampling is by itself not differentiable, so in order to integrate it with autograd,
    this implementation follows the paper
    "Estimating or propagating gradients through stochastic neurons for conditional computation"
    and simply returns the Bernoulli probabilities themselves as the "gradient". This is called the
    "straight-through estimator". For more details, please see Section 4 of the aforementioned paper
    (https://arxiv.org/pdf/1308.3432).
    """

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

    # This method determines the "gradient" of the BernoulliSample function.
    # grad_output is supposed to be the gradient w.r.t. the output of the forward method.
    @staticmethod
    def backward(ctx: torch.Any, grad_output: torch.Tensor) -> torch.Tensor:  # type: ignore
        # ctx.saved_tensors is a tuple (of length 1 in this case). Hence the indexing here.
        bernoulli_probs = ctx.saved_tensors[0]
        return bernoulli_probs * grad_output


bernoulli_sample = BernoulliSample.apply


def sigmoid_inverse(x: torch.Tensor) -> torch.Tensor:
    return -torch.log(1 / x - 1)
