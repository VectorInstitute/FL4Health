from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
import torch.nn as nn

from fl4health.utils.dataset_converter import AutoEncoderDatasetConverter


class AbstractAe(nn.Module, ABC):
    """The base class for all Encoder-Decoder based models.
    All we need to define such model is the type of the model, and the structure
    of the encoder and the decoder modules. This type of model should have the capability
    to encode data using the encoder module and decode the output of the encoder using the decoder module.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
    ) -> None:
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
    """Variational Auto-Encoder model base class."""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
    ) -> None:
        """Initializes the VariationalAE class.

        Args:
            model_type (AutoEncoderType): The model type should be VARIATIONAL_AE.
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
        return eps.mul(std).add_(mu)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        mu, logvar = self.encode(input)
        z = self.sampling(mu, logvar)
        output = self.decode(z)
        # Output (reconstruction) is flattened to be concatenated with mu and logvar vectors.
        # The shape of the flattened_output can be later restored by having the training data shape,
        # or the decoder structure.
        flattened_output = output.view(output.shape[0], -1)
        return torch.cat((logvar, mu, flattened_output), dim=1)


class ConditionalVae(AbstractAe):
    def __init__(
        self, encoder: nn.Module, decoder: nn.Module, converter: Optional[AutoEncoderDatasetConverter] = None
    ) -> None:
        """Conditional Variatioan Auto-Encoder model.

        Args:
            encoder (nn.Module): The encoder used to map input to latent space.
            decoder (nn.Module): The decoder used to reconstruct the input using a vector in latent space.

        Returns:
            _type_: _description_
        """

        super().__init__(encoder, decoder)
        self.converter = converter

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
        return eps.mul(std).add_(mu)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert self.converter is not None
        input, condition = self.converter.unpack_input_condition(input)
        # print("In forward input", input.shape)
        # print("In forward condition", condition.shape)
        mu, logvar = self.encode(input, condition)
        z = self.sampling(mu, logvar)
        output = self.decode(z, condition)
        # Output (reconstruction) is flattened to be concatenated with mu and logvar vectors.
        # The shape of the flattened_output can be later restored by having the training data shape,
        # or the decoder structure.
        # print("output size", output.shape)
        # print("mu and logvar size", mu.shape, logvar.shape)
        flattened_output = output.view(output.shape[0], -1)
        return torch.cat((logvar, mu, flattened_output), dim=1)
