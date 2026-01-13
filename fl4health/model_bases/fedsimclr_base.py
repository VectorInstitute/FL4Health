from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


DEFAULT_PROJECTION_HEAD = nn.Identity()


class FedSimClrModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        projection_head: nn.Module = DEFAULT_PROJECTION_HEAD,
        prediction_head: nn.Module | None = None,
        pretrain: bool = True,
    ) -> None:
        """
        Model base to train SimCLR (https://arxiv.org/pdf/2002.05709)
        in a federated manner presented in (https://arxiv.org/pdf/2207.09158).
        Can be used in pretraining and optionally finetuning.

        Args:
            encoder (nn.Module): Encoder that extracts a feature vector. given an input sample.
            projection_head (nn.Module): Projection Head that maps output of encoder to final representation used in
                contrastive loss for pretraining stage. Defaults to identity transformation.
            prediction_head (nn.Module | None): Prediction head that maps output of encoder to prediction in the
                finetuning stage. Defaults to None.
            pretrain (bool): Determines whether or not to use the ``projection_head`` (True) or the
                ``prediction_head`` (False). Defaults to True.
        """
        super().__init__()

        assert not (prediction_head is None and not pretrain), (
            "Model with pretrain==False must have prediction head (ie not None)"
        )

        self.encoder = encoder
        self.projection_head = projection_head
        self.prediction_head = prediction_head
        self.pretrain = pretrain

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Passes the input tensor through the encoder module. If we're in the pretraining phase, the output of the
        encoder is flattened/projected for similarity computations. If we're fine-tuning the model, these latent
        features are passes through the provided prediction head.

        Args:
            input (torch.Tensor): Input to be mapped to either latent features or a final prediction depending on the
                training phase.

        Returns:
            (torch.Tensor): The output from either the ``projection_head`` module if pre-training or the
                ``prediction_head`` if fine-tuning.
        """
        features = self.encoder(input)
        if self.pretrain:
            return self.projection_head(features)
        assert self.prediction_head is not None, "Model with pretrain==False must have prediction_head (ie not None)"
        return self.prediction_head(features)

    @staticmethod
    def load_pretrained_model(model_path: Path) -> FedSimClrModel:
        """
        Given a path, this function loads a model from the path, assuming was of type ``FedSimClrModel``. The proper
        components are then routed to form a new model with the pre-existing weights.

        **NOTE**: Loaded models automatically set ``pretrain`` to False

        Args:
            model_path (Path): Path to a ``FedSimClrModel`` object saved using ``torch.save``

        Returns:
            (FedSimClrModel): A model with pre-existing weights loaded and ``pretrain`` set to False
        """
        prev_model = torch.load(model_path, weights_only=False)
        return FedSimClrModel(
            encoder=prev_model.encoder,
            projection_head=prev_model.projection_head,
            prediction_head=prev_model.prediction_head,
            pretrain=False,
        )
