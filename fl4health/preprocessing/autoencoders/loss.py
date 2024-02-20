from typing import Tuple

import torch
from torch.nn.modules.loss import _Loss


class VaeLoss(_Loss):
    def __init__(
        self,
        latent_dim: int,
        base_loss: _Loss,
    ) -> None:
        """The loss function used for training CVAEs and VAEs.
        This loss computes the base_loss (defined by the user) between the input and generated output.
        It then adds the KL divergence between the estimated distribution (represented by mu and logvar)
        and the standard normal distribution.

        Args:
            latent_dim (int): Dimensionality of the latent space.
            base_loss (_Loss): Base loss function between the input and reconstruction.
        """
        super().__init__()
        # User can define a base_loss to measure the distance between the input and generated output.
        self.base_loss = base_loss
        # Latent dimension is used to unpack the model output
        self.latent_dim = latent_dim

    def standard_normal_kl_divergence_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Calculates the analytical KL divergence between the normal distribution
        and the estimated distribution.

        Args:
            mu (torch.Tensor): Mean of the estimated distribution.
            logvar (torch.Tensor): Log variance of the estimated distribution.

        Returns:
            torch.Tensor: KL divergence loss.
        """
        kl_divergence_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return kl_divergence_loss

    def unpack_model_output(self, preds: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Unpacks the model output tensor.

        Args:
            preds (torch.Tensor): Model predictions.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Unpacked output containing predictions, mu, and logvar.
        """
        # This methods assumes "preds" are batch first, and preds are 2D dimensional (already flattened).
        assert (
            preds.dim() == 2
        ), f"Expected a 2D tensor for VaeLoss, but got {preds.dim()}D tensor with shape {preds.shape}."
        # The order of logvar and mu in the output tensor is important.
        # For each model output, the first self.latent_dim indices are used to store the log variance,
        # the next self.latent_dim indices are allocated to mu, and the remaining indices store the model predictions.
        logvar = preds[:, 0 : self.latent_dim]
        mu = preds[:, self.latent_dim : 2 * self.latent_dim]
        preds = preds[:, 2 * self.latent_dim :]
        return preds, mu, logvar

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculates the total loss.

        Args:
            preds (torch.Tensor): Model predictions.
            target (torch.Tensor): Target values.

        Returns:
            torch.Tensor: Total loss composed of base loss and KL divergence loss.
        """
        flattened_output, mu, logvar = self.unpack_model_output(preds)
        kl_loss = self.standard_normal_kl_divergence_loss(mu, logvar)
        # Reshaping the flattened output to its original shape.
        return self.base_loss(flattened_output.view(*target.shape), target) + kl_loss
