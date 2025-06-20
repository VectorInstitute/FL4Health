import torch
from flwr.common.typing import Config, NDArrays
from torch import nn

from fl4health.parameter_exchange.layer_exchanger import DynamicLayerExchanger
from fl4health.parameter_exchange.parameter_selection_criteria import select_scores_and_sample_masks
from fl4health.utils.functions import sigmoid_inverse


class FedPmExchanger(DynamicLayerExchanger):
    def __init__(self) -> None:
        """
        Exchanger specifically tailored to exchange parameters and other information between FedPM servers and
        clients. FedPM has a special set of information that needs to be exchanged, which is handled by this class.
        """
        super().__init__(select_scores_and_sample_masks)

    def pull_parameters(self, parameters: NDArrays, model: nn.Module, config: Config | None = None) -> None:
        current_state = model.state_dict()
        layer_params, layer_names = self.unpack_parameters(parameters)
        for layer_name, layer_param in zip(layer_names, layer_params):
            # Apply the inverse of the Sigmoid function
            # since the scores for masked layers are supposed to be unbounded.
            with torch.no_grad():
                current_state[layer_name] = sigmoid_inverse(torch.tensor(layer_param))
        model.load_state_dict(current_state, strict=True)
