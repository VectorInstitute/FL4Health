from abc import abstractmethod
from typing import Generic, TypeVar

from flwr.common.typing import NDArrays
from torch import nn

from fl4health.parameter_exchange.parameter_exchanger_base import ParameterExchanger
from fl4health.parameter_exchange.parameter_packer import ParameterPacker


T = TypeVar("T")


class PartialParameterExchanger(ParameterExchanger, Generic[T]):
    def __init__(self, parameter_packer: ParameterPacker[T]) -> None:
        """
        Base class meant to properly facilitate partial parameter exchange through a selection criterion. This
        mechanism is more complicated than, for example, that used by the ``FixedLayerExchanger`` where the subset
        parameters to exchange do not change dynamically from round to round.

        Args:
            parameter_packer (ParameterPacker[T]): Parameter packer that can be used to pack in more information
                than just the parameters being exchange. This is important, for example, when exchanging different
                sets of layers in each round.
        """
        super().__init__()
        self.parameter_packer = parameter_packer

    def pack_parameters(self, model_weights: NDArrays, additional_parameters: T) -> NDArrays:
        return self.parameter_packer.pack_parameters(model_weights, additional_parameters)

    def unpack_parameters(self, packed_parameters: NDArrays) -> tuple[NDArrays, T]:
        return self.parameter_packer.unpack_parameters(packed_parameters)

    @abstractmethod
    def select_parameters(
        self,
        model: nn.Module,
        initial_model: nn.Module | None = None,
    ) -> tuple[NDArrays, T]:
        raise NotImplementedError
