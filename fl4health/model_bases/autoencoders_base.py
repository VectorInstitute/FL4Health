from abc import ABC, abstractmethod
from enum import Enum
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoEncoderType(Enum):
    BASIC_AE = "BASIC_AE"
    VARIATIONAL_AE = "VARIATIONAL_AE"
    CONDITIONAL_VAE = "CONDITIONAL_VAE"


class AbstractAE(nn.Module, ABC):
    """The base class for all Encoder-Decoder based models.
    All we need to define such model is the type of the model, and the structure
    of the encoder and the decoder modules. This type of model should have the capability
    to encode data using the encoder module and decode the output of the encoder using the decoder module.
    """

    def __init__(
        self,
        model_type: AutoEncoderType,
        encoder: nn.Module,
        decoder: nn.Module,
    ) -> None:
        super().__init__()
        self.model_type = model_type
        self.encoder = encoder
        self.decoder = decoder

    @abstractmethod
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class VariationalAE(AbstractAE):
    """Variational Auto-Encoder model base class."""

    def __init__(
        self,
        model_type: AutoEncoderType,
        encoder: nn.Module,
        decoder: nn.Module,
    ) -> None:
        """Initializes the VariationalAE class.

        Args:
            model_type (AutoEncoderType): The model type should be VARIATIONAL_AE.
            encoder (nn.Module): Encoder module defined by the user.
            decoder (nn.Module): Decoder module defined by the user.
        """
        super().__init__(model_type, encoder, decoder)

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
        return torch.cat((logvar, mu, output), dim=1)


class ConditionalVAE(AbstractAE):
    def __init__(
        self,
        model_type: AutoEncoderType,
        num_conditions: int,
        encoder: nn.Module,
        decoder: nn.Module,
    ) -> None:
        """Initializes the ConditionalVAE class.

        Args:
            model_type (AutoEncoderType): The model type should be CONDITIONAL_VAE.
            num_conditions (int): The total number of conditions, which is used to prepare the condition vector.
            encoder (nn.Module): Encoder module defined by the user.
            decoder (nn.Module): Decoder module defined by the user.
        """
        super().__init__(model_type, encoder, decoder)
        self.num_conditions = num_conditions

    def cat_input_condition(self, input: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        input_cond = torch.cat((input, condition), dim=-1)
        return input_cond

    def one_hot(self, condition: torch.Tensor) -> torch.Tensor:
        # One-hot encoding of the condition
        condition = F.one_hot(condition.to(torch.int64), self.num_conditions).to(condition.device)
        return condition

    def maybe_reshape(self, condition_matrix: torch.Tensor) -> torch.Tensor:
        # Make sure the condition has the right shape.
        if len(condition_matrix.shape) > 2:
            # Since the condition is replicated to match the size of the input,
            #  we need to narrow it back just to one number per sample
            return condition_matrix[:, 0, 0]
        else:
            # Condition was not replicated to match the size of the input.
            return condition_matrix

    def encode(self, input: torch.Tensor, condition: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # User can decide how to use the condition in the encoder,
        # ex: using the condition in the middle layers of encoder.
        mu, logvar = self.encoder(input, condition)
        return mu, logvar

    def decode(self, latent_vector: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        latent_vector = self.cat_input_condition(latent_vector, condition)
        output = self.decoder(latent_vector)
        return output

    def sampling(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input[:, :-1]  # Exclude the last item (conditions)
        condition_matrix = input[:, -1]  # Last item contains conditions
        condition = self.maybe_reshape(condition_matrix)
        one_hot_condition = self.one_hot(condition)
        mu, logvar = self.encode(x, one_hot_condition)
        z = self.sampling(mu, logvar)
        output = self.decode(z, one_hot_condition)
        # The shape of the flattened_output can be later restored by having the training image shape,
        # or the decoder structure.
        flattened_output = output.view(output.shape[0], -1)
        return torch.cat((logvar, mu, flattened_output), dim=1)


class BasicAE(AbstractAE):
    def __init__(self, model_type: AutoEncoderType, encoder: nn.Module, decoder: nn.Module) -> None:
        super().__init__(model_type, encoder, decoder)

    def encode(self, input: torch.Tensor) -> torch.Tensor:
        latent_vector = self.encoder(input)
        return latent_vector

    def decode(self, latent_vector: torch.Tensor) -> torch.Tensor:
        output = self.decoder(latent_vector)
        return output

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        z = self.encode(input)
        return self.decode(z)