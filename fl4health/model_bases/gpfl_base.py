from logging import WARNING

import torch
import torch.nn.functional as F
from flwr.common.logger import log
from torch import nn

from fl4health.model_bases.partial_layer_exchange_model import PartialLayerExchangeModel
from fl4health.model_bases.sequential_split_models import SequentiallySplitExchangeBaseModel


class Gce(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int) -> None:
        """
        Taken from the official implementation at : https://github.com/TsingZ0/GPFL/blob/main/system/flcore/servers/servergp.py
        GCE module as described in the GPFL paper. This module is used as a lookup table of global class embeddings.
        The size of the embedding matrix (the lookup table) is (num_classes, feature_dim). The goal is to learn
        and store representative class embeddings.

        Args:
            feature_dim (int): The dimension of the feature tensor.
            num_classes (int): The number of classes represented in the embedding table.
        """
        super(Gce, self).__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.embedding = nn.Embedding(num_classes, feature_dim)

    def forward(self, feature_tensor: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the GCE module. It computes the cosine similarity between the feature tensors
        and the class embeddings, and then computes the log softmax loss based on the provided labels.

        Args:
            feature_tensor (torch.Tensor): The global features computed by the CoV module.
            label (torch.Tensor): The true label for the input data, which is used to compute the loss.

        Returns:
            (torch.Tensor): Log softmax loss.
        """
        # Invoke the forward of the embedding layer to make sure the computation graph is connected
        # and embedding parameters are updated during the backward pass.
        embeddings = self.embedding(torch.tensor(range(self.num_classes)))
        # We are computing the dot product using F.Linear.
        cosine = F.linear(F.normalize(feature_tensor), F.normalize(embeddings))
        if label.dim() == 1:
            one_hot = torch.zeros(cosine.size())
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        else:
            assert label.shape[1] == self.num_classes, (
                "Shape of the one-hot encoded labels should be (batch_size, num_classes)."
            )
            # Label is already one-hot encoded with the shape of (batch_size, num_classes)
            one_hot = label

        softmax_value = F.log_softmax(cosine, dim=1)
        softmax_loss = one_hot * softmax_value
        return -torch.mean(torch.sum(softmax_loss, dim=1))

    def lookup(self, target: torch.Tensor) -> torch.Tensor:
        """
        Extracts the class embeddings for the given target vectors.

        Args:
            target (torch.Tensor): A tensor containing the indices or one-hot embeddings
                of the classes to look up.

        Returns:
            (torch.Tensor): The class embeddings corresponding to the provided targets.
        """
        if self.training:
            log(
                WARNING,
                "Lookup method should not be used for training. "
                "This method is intended for the purpose of embedding lookup, and "
                "does not invoke the forward pass.",
            )
        one_hot_n_dim = 2  # To avoid having magic numbers
        if target.dim() == one_hot_n_dim:
            assert target.shape[1] == self.num_classes, (
                "Shape of the one-hot encoded labels should be (batch_size, num_classes)."
            )
            # If the target is one-hot encoded, convert it to indices.
            target = torch.argmax(target, dim=1)

        assert target.shape == (target.shape[0],), "lookup requires 1D tensor of class indices."
        return self.embedding.weight.data[target.int()]


class CoV(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        """
        Taken from the official implementation at : https://github.com/TsingZ0/GPFL/blob/main/system/flcore/servers/servergp.py
        CoV (Conditional Value) module as described in the GPFL paper. This module consists of two parts.
        1) First, uses the provided context tensor to compute two vectors, \\(\\gamma\\) and \\(\\beta\\) using
        ``conditional_gamma`` and ``conditional_beta`` sub-modules, respectively.
        In the paper: \\([\\mathbf{\\gamma_i}, \\mathbf{\\beta_i} = \\text{CoV}(\\mathbf{f}_i, \\cdot, V)]\\)
        2) Then, applies an affine transformation followed by a ReLU activation to the feature tensors based on
        the computed \\(\\gamma\\) and \\(\\beta\\) vectors.
        Affine transformation in the paper:
        \\([(\\mathbf{\\gamma} + \\mathbf{1})\\odot \\mathbf{f}_i + \\mathbf{\\beta}]\\)
        Parameters of the sub-modules (``conditional_gamma`` and ``conditional_beta`` modules) are the main
        components of this module, and are optimized during the training process.

        Args:
            feature_dim (int): The dimension of the feature tensor.
        """
        super(CoV, self).__init__()
        self.conditional_gamma = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.LayerNorm([feature_dim]),
        )
        self.conditional_beta = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.LayerNorm([feature_dim]),
        )
        self.activation = nn.ReLU()

    def forward(self, feature_tensor: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Uses the context tensor to compute gamma and beta vectors. Then, applies a conditional
        affine transformation to the feature tensor based on the computed gamma and beta vectors.

        Args:
            feature_tensor (torch.Tensor): Output of the base feature extractor.
            context (torch.Tensor): The conditional tensor that could be global or personalized.

        Returns:
            (torch.Tensor): The transformed feature tensor after applying the conditional affine transformation.
        """
        # Call submodules to compute gamma and beta vectors.
        gamma = self.conditional_gamma(context)
        beta = self.conditional_beta(context)

        # Now do the affine transformation with gamma and beta vectors.
        out = torch.multiply(feature_tensor, gamma + 1)
        out = torch.add(out, beta)
        return self.activation(out)


class GpflBaseAndHeadModules(SequentiallySplitExchangeBaseModel):
    def __init__(self, base_module: nn.Module, head_module: nn.Module, flatten_features: bool) -> None:
        """
        This module class holds the main components for prediction in the GPFL model.
        This is mainly used to enable defining one optimizer for the base and head modules.

        Args:
            base_module (nn.Module): Base feature extractor module that generates a feature tensor from the input.
            head_module (nn.Module): Head module that takes a personalized feature tensor and produces the
                final predictions.
            flatten_features (bool): Whether the ``base_module``'s output features should be flattened or not.
        """
        super().__init__(base_module=base_module, head_module=head_module, flatten_features=flatten_features)

    def forward(self, input: torch.Tensor) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """
        A wrapper around the default sequential forward pass of the GPFL model base to restrict its usage.

        Args:
            input (torch.Tensor): Input to the model forward pass.

        Returns:
            (tuple[torch.Tensor, torch.Tensor]): Return the prediction dictionary and a features dictionaries.
        """
        # Throw an error because this function should not directly be called with this class.
        raise NotImplementedError("Forward pass should not be used for the GpflBaseAndHeadModules class. ")


class GpflModel(PartialLayerExchangeModel):
    def __init__(
        self,
        base_module: nn.Module,
        head_module: nn.Module,
        feature_dim: int,
        num_classes: int,
        flatten_features: bool = False,
    ) -> None:
        """
        GPFL model base as described in the paper "GPFL: Simultaneously Learning Global and Personalized
        Feature Information for Personalized Federated Learning." https://arxiv.org/abs/2308.10279
        This base module consists of three main sub-modules: the main_module, which consists of
        a feature extractor and a head module; the GCE (Global Conditional Embedding) module; and
        the CoV (Conditional Value) module.

        Args:
            base_module (nn.Module): Base feature extractor module that generates a feature tensor from the input.
            head_module (nn.Module): Head module that takes a personalized feature tensor and produces the
                final predictions.
            feature_dim (int): The output dimension of the base feature extractor. This is also the input dimension
                of the head and CoV modules.
            num_classes (int): This is used to construct the GCE module.
            flatten_features (bool, optional): Whether the ``base_module``'s output features should be
                flattened or not. Defaults to False.
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.gpfl_main_module = GpflBaseAndHeadModules(base_module, head_module, flatten_features)
        self.cov = CoV(feature_dim)
        self.gce = Gce(feature_dim, num_classes)

    def forward(
        self,
        input: torch.Tensor,
        global_conditional_input: torch.Tensor,
        personalized_conditional_input: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """
        There are two types of forward passes in this model base. The first is the forward pass preformed
        during training.
        During training:
        1) Input is passed through the base feature extractor.
        2) Then the CoV module maps the extracted features into two feature tensors corresponding to local and global
           features. The CoV module requires ``global_conditional_input`` and ``personalized_conditional_input``
           tensors, which are used to condition the output of the CoV module. These tensors are computed in clients at
           the beginning of each round.
        3) The ``local_features`` are fed into the ``head_module`` to produce class predictions.
        4) The ``global_conditional_input`` is used to compute the global features, and these ``global_features`` to
           be used in loss calculations and are returned only during training.

        The second type of forward pass happens during evaluation. For evaluation:

        1) Input is passed through the base feature extractor.
        2) ``local_features`` are generated by the CoV module.
        3) These local features are passed through the head module to produce the final predictions.

        Args:
            input (torch.Tensor): Input tensor to be fed into the feature extractor.
            global_conditional_input (torch.Tensor): The conditional input tensor used by the CoV module
                to generate the global features.
            personalized_conditional_input (torch.Tensor): The conditional input tensor used by the CoV module
                to generate the local features.

        Returns:
            (tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]): A tuple in which the first element
                contains a dictionary of predictions and the second element contains intermediate features
                indexed by name.
        """
        # Pass the input through the base feature extractor and potentially flatten the features.
        features = self.gpfl_main_module.features_forward(input)
        assert features.shape[1] == self.feature_dim, (
            "Feature dimension mismatch between output of the base module and the expected feature_dim by CoV."
        )
        local_features = self.cov(features, personalized_conditional_input)
        assert local_features.shape[1] == self.feature_dim, (
            "Local feature dimension mismatch between output of the CoV module "
            "and the expected feature_dim by the head module."
        )
        predictions = self.gpfl_main_module.head_module.forward(local_features)
        if not self.training:
            return {"prediction": predictions}, {}

        assert len(global_conditional_input) == self.feature_dim, (
            "global_conditional_input must match the expected feature dimension by the CoV module."
        )
        global_features = self.cov(features, global_conditional_input)
        assert global_features.shape[1] == self.feature_dim, "global_features dimension should match the feature_dim."
        return {"prediction": predictions}, {"local_features": local_features, "global_features": global_features}

    def layers_to_exchange(self) -> list[str]:
        """
        Returns a list of layer names that should be exchanged between the server and clients.

        Returns:
            (list[str]): A list of layer names that should be exchanged. This is used by the
                ``FixedLayerExchanger`` class to determine which layers to exchange during the FL process.
        """
        base_layers = self.gpfl_main_module.layers_to_exchange()
        # gpfl_main_module's layers_to_exchange returns base module layers starting with "base_module."
        # We need to prepend "gpfl_main_module." to these layer names to match the state_dict keys.
        complete_base_layer_names = [f"gpfl_main_module.{layer_name}" for layer_name in base_layers]
        gpfl_module_layers = [
            layer_name for layer_name in self.state_dict() if layer_name.startswith(("cov.", "gce."))
        ]
        return complete_base_layer_names + gpfl_module_layers
