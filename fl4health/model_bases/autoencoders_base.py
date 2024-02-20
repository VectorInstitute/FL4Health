from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn


class AbstractAe(nn.Module, ABC):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
    ) -> None:
        """
        The base class for all autoencoder based models.
        To define this model, we need to define the structure of the encoder and the decoder modules.
        This type of model should have the capability to encode data using the encoder module and decode
        the output of the encoder using the decoder module.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    @abstractmethod
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Forward is called in client classes with a single input tensor.
        raise NotImplementedError


class BasicAe(AbstractAe):
    def __init__(self, encoder: nn.Module, decoder: nn.Module) -> None:
        super().__init__(encoder, decoder)

    def encode(self, input: torch.Tensor) -> torch.Tensor:
        latent_vector = self.encoder(input)
        return latent_vector

    def decode(self, latent_vector: torch.Tensor) -> torch.Tensor:
        output = self.decoder(latent_vector)
        return output

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        z = self.encode(input)
        return self.decode(z)


class VariationalAe(AbstractAe):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
    ) -> None:
        """Variational Auto-Encoder model base class.

        Args:
            encoder (nn.Module): Encoder module defined by the user.
            decoder (nn.Module): Decoder module defined by the user.
        """
        super().__init__(encoder, decoder)

    def encode(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(input)
        return mu, logvar

    def decode(self, latent_vector: torch.Tensor) -> torch.Tensor:
        output = self.decoder(latent_vector)
        return output

    def sampling(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        mu, logvar = self.encode(input)
        z = self.sampling(mu, logvar)
        output = self.decode(z)
        # Output (reconstruction) is flattened to be concatenated with mu and logvar vectors.
        # The shape of the flattened_output can be later restored by having the training data shape,
        # or the decoder structure.
        # This assumes output is "batch first".
        flattened_output = output.view(output.shape[0], -1)
        return torch.cat((logvar, mu, flattened_output), dim=1)


class ConditionalVae(AbstractAe):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        unpack_input_condition: Optional[Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> None:
        """Conditional Variational Auto-Encoder model.

        Args:
            encoder (nn.Module): The encoder used to map input to latent space.
            decoder (nn.Module): The decoder used to reconstruct the input using a vector in latent space.
            unpack_input_condition (Optional[Callable], optional): For unpacking the input and condition tensors.
        """

        super().__init__(encoder, decoder)
        self.unpack_input_condition = unpack_input_condition

    def encode(
        self, input: torch.Tensor, condition: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # User can decide how to use the condition in the encoder,
        # ex: using the condition in the middle layers of encoder.
        mu, logvar = self.encoder(input, condition)
        return mu, logvar

    def decode(self, latent_vector: torch.Tensor, condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        # User can decide how to use the condition in the decoder,
        # ex: using the condition in the middle layers of decoder, or not using it at all.
        output = self.decoder(latent_vector, condition)
        return output

    def sampling(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert self.unpack_input_condition is not None
        input, condition = self.unpack_input_condition(input)
        mu, logvar = self.encode(input, condition)
        z = self.sampling(mu, logvar)
        output = self.decode(z, condition)
        # Output (reconstruction) is flattened to be concatenated with mu and logvar vectors.
        # The shape of the flattened_output can be later restored by having the training data shape,
        # or the decoder structure.
        # This assumes output is "batch first".
        flattened_output = output.view(output.shape[0], -1)
        return torch.cat((logvar, mu, flattened_output), dim=1)
