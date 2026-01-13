from collections import OrderedDict

import torch
from flwr.common.typing import Config, NDArrays
from torch import nn

from fl4health.parameter_exchange.parameter_exchanger_base import ParameterExchanger


class FullParameterExchanger(ParameterExchanger):
    def push_parameters(
        self, model: nn.Module, initial_model: nn.Module | None = None, config: Config | None = None
    ) -> NDArrays:
        """
        Sending all of parameters ordered by ``state_dict`` keys.

        **NOTE**: Order matters, because it is relied upon by ``pull_parameters`` below.

        Args:
            model (nn.Module): Model containing the weights to be sent.
            initial_model (nn.Module | None, optional): Not Used. Defaults to None.
            config (Config | None, optional): Not Used. Defaults to None.

        Returns:
            (NDArrays): All parameters contained in the ``state_dict`` of the model parameter. The ``state_dict``
                maintains a specific order.
        """
        # Sending all of parameters ordered by state_dict keys
        # NOTE: Order matters, because it is relied upon by pull_parameters below
        return [val.cpu().numpy() for _, val in model.state_dict().items()]

    def pull_parameters(self, parameters: NDArrays, model: nn.Module, config: Config | None = None) -> None:
        """
        Takes in a set of parameters in the form of ``NDArrays`` (list of numpy arrays) and injects them into the
        provided model.

        Assumes all model parameters are contained in parameters. The ``state_dict`` is reconstituted because
        parameters is simply a list of arrays.

        Args:
            parameters (NDArrays): Parameter to inject into the provided model.
            model (nn.Module): Model to inject the parameters into.
            config (Config | None, optional): Not used. Defaults to None.
        """
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
