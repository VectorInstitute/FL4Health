from collections import OrderedDict
from typing import Tuple, List

from torch import Size, Tensor, cat, flatten, float64, nn, prod, tensor
import torch 
from torch.linalg import vector_norm
from flwr.common.logger import log
from logging import INFO


def get_model_dimension(model: nn.Module) -> int:
    return sum(p.numel() for p in model.state_dict().values())


def tensor_shape_to_parameter_count(tensor_shape: Size) -> int:
    """Converts tensor shape to number of tensor components."""
    return prod(tensor(list(tensor_shape))).item()


def vectorize_model(model: nn.Module) -> Tensor:
    """Get model vector by vectorizing each layer and concatenating them."""
    vectorized_layers = [flatten(layer) for layer in model.state_dict().values()]
    return cat(tuple(vectorized_layers))


def unvectorize_model(model: nn.Module, parameter_vector: Tensor) -> nn.Module:
    """Hydrate model from parameter vector."""
    assert parameter_vector.dim() == 1  # ensure we received a vector
    assert get_model_dimension(model) == parameter_vector.numel()  # ensure dimensions agree

    start = end = 0
    state_dict = model.state_dict()
    for layer_name, layer_params in state_dict.items():
        shape = layer_params.size()
        end += layer_params.numel()
        # if layer_params.numel() == 1:
        #     log(INFO, layer_name)
        #     log(INFO, layer_params)
        #     log(INFO, get_model_dimension(model))
        # log(INFO, f'start: {start}, end {end}, shape {shape}, numel: {layer_params.numel()}')
        segment = parameter_vector[start:end]
        state_dict[layer_name] = segment.view(shape)    
        start = end

    model.load_state_dict(state_dict)
    return model


def get_model_norm(model: nn.Module, p=2) -> Tensor:
    """Finds the p-norm of the model."""
    vect = vectorize_model(model).to(float64)
    return vector_norm(vect, ord=p)

def get_model_layer_types(model: nn.Module) -> List[torch.dtype]:
    return [layer.dtype for layer in model.state_dict().values()]

def change_model_dtypes(model: nn.Module, dtypes_list: List[torch.dtype]) -> nn.Module:

    model_state_dict = model.state_dict()

    assert len(model_state_dict) == len(dtypes_list)

    i = 0
    for key in model_state_dict.keys():
        model_state_dict[key] = model_state_dict[key].to(dtypes_list[i])
        i += 1

    model.load_state_dict(model_state_dict)
    return model
