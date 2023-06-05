from typing import Tuple, Union

import numpy as np
from flwr.common.typing import List, NDArrays

from fl4health.parameter_exchange.full_exchanger import FullParameterExchanger


class ParameterExchangerWithPacking(FullParameterExchanger):
    def pack_parameters(self, model_weights: NDArrays, additional_parameters: Union[NDArrays, float]) -> NDArrays:
        if isinstance(additional_parameters, float):
            return model_weights + [np.array(additional_parameters)]
        else:
            return model_weights + additional_parameters


class ParameterExchangerWithControlVariates(ParameterExchangerWithPacking):
    def unpack_parameters(self, packed_parameters: NDArrays) -> Tuple[NDArrays, NDArrays]:
        # Ensure that the packed parameters is even as a sanity check. Model paramers and control variates have same
        # size.
        assert len(packed_parameters) % 2 == 0
        split_size = len(packed_parameters) // 2
        return packed_parameters[:split_size], packed_parameters[split_size:]


class ParameterExchangerWithClippingBit(ParameterExchangerWithPacking):
    def unpack_parameters(self, packed_parameters: NDArrays) -> Tuple[NDArrays, float]:
        # The last entry in the parameters list is assumed to be a clipping bound (even if we're evaluating)
        split_size = len(packed_parameters) - 1
        model_parameters = packed_parameters[:split_size]
        clipping_bound = float(packed_parameters[split_size:][0])
        return model_parameters, clipping_bound


class ParameterExchangerWithNames(ParameterExchangerWithPacking):
    def pack_parameters(self, model_weights: NDArrays, weights_names: Union[NDArrays, float, List[str]]) -> NDArrays:
        return model_weights + [np.array(weights_names)]

    def unpack_parameters(self, packed_parameters: NDArrays) -> Tuple[NDArrays, List[str]]:
        split_size = len(packed_parameters) - 1
        model_parameters = packed_parameters[:split_size]
        param_names = packed_parameters[split_size:][0].tolist()
        return model_parameters, param_names
