from abc import ABC, abstractmethod
from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F

from fl4health.clients.basic_client import BasicClient
from fl4health.utils.losses import Losses
from fl4health.utils.dataset import BaseDataset


class AutoEncoderType(Enum):
    BASIC_AE = "BASIC_AE"
    VARIATIONAL_AE = "VARIATIONAL_AE"
    CONDITIONAL_VAE = "CONDITIONAL_VAE"


class AutoEncoderBase(nn.Module, ABC):
    """The base class for all Encoder-Decoder based models.
    All we need to define such model is the type of the model, and the structure of the encoder and the decoder modules.
    This type of model should have the capability to encode data using the encoder module and decode the output of the encoder using the decoder module.
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
    def encode() -> torch.Tensor():
        raise NotImplementedError

    @abstractmethod
    def decode() -> torch.Tensor():
        raise NotImplementedError

    @abstractmethod
    def forward() -> torch.Tensor():
        raise NotImplementedError


class VarioationalAE(AutoEncoderBase):
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

    def encode(self, input: torch.Tensor) -> torch.Tensor:
        mu, logvar = self.encoder(input)
        return mu, logvar

    def decode(self, latent_vector: torch.Tensor) -> torch.Tensor:
        output = self.decoder(latent_vector)
        return output

    def sampling(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, input):
        mu, logvar = self.encode(input)
        z = self.sampling(mu, logvar)
        output = self.decode(z)
        return torch.cat((logvar, mu, output), dim=1)


class ConditionalVAE(AutoEncoderBase):
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

    def encode(self, input: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        input = self.cat_input_condition(input, condition)
        mu, logvar = self.encoder(input)
        return mu, logvar

    def decode(self, latent_vector: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        latent_vector = self.cat_input_condition(latent_vector, condition)
        output = self.decoder(latent_vector)
        return output

    def sampling(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, input):
        x = input[:, :-1]  # Exclude the last column (conditions)
        condition = input[:, -1]  # Last column contains conditions
        condition = self.one_hot(condition)
        mu, logvar = self.encode(x, condition)
        z = self.sampling(mu, logvar)
        output = self.decode(z, condition)
        return torch.cat((logvar, mu, output), dim=1)


class Basic_AE(AutoEncoderBase):
    def __init__(self, model_type: AutoEncoderType, encoder: nn.Module, decoder: nn.Module) -> None:
        super().__init__(model_type, encoder, decoder)

    def encode(self, input: torch.Tensor) -> torch.Tensor:
        latent_vector = self.encoder(input)
        return latent_vector

    def decode(self, latent_vector: torch.Tensor) -> torch.Tensor:
        output = self.decoder(latent_vector)
        return output

    def forward(self, input):
        z = self.encode(input)
        return self.decode(z)
