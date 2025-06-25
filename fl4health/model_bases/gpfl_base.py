import torch
import torch.nn.functional as F
from torch import nn

from fl4health.model_bases.partial_layer_exchange_model import PartialLayerExchangeModel


class GCE(nn.Module):
    # Taken from the official implementation at : https://github.com/TsingZ0/GPFL/blob/main/system/flcore/servers/servergp.py
    def __init__(self, feature_dim: int, num_classes: int) -> None:
        """
        GCE module as described in the GPFL paper. This module is used as a lookup table of global class embeddings.
        The size of the embedding matrix (the lookup table) is (num_classes, feature_dim). The goal is to learn
        and store representative class embeddings.

        Args:
            feature_dim (int): The dimension of the feature tensor.
            num_classes (int): The number of classes represented in the embedding table.
        """
        super(GCE, self).__init__()
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
            torch.Tensor: Log softmax loss.
        """
        embeddings = self.embedding(torch.tensor(range(self.num_classes)))
        cosine = F.linear(F.normalize(feature_tensor), F.normalize(embeddings))
        one_hot = torch.zeros(cosine.size())
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        softmax_value = F.log_softmax(cosine, dim=1)
        softmax_loss = one_hot * softmax_value
        return -torch.mean(torch.sum(softmax_loss, dim=1))


class CoV(nn.Module):
    # Taken from the official implementation at : https://github.com/TsingZ0/GPFL/blob/main/system/flcore/servers/servergp.py
    def __init__(self, feature_dim: int) -> None:
        """
        CoV (Conditional Value) module as described in the GPFL paper. This module takes a feature tensor and
        a context tensor, and applies an affine transformation to the feature tensor based on the context.
        Parameters of the affine transformation are the main components of this module, and are optimized
        during the training process.

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
        self.act = nn.ReLU()

    def forward(self, feature_tensor: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Applies the conditional affine transformation to the feature tensor based on the context tensor.

        Args:
            feature_tensor (torch.Tensor): Output of the base feature extractor.
            context (torch.Tensor): The conditional tensor that could be generic or personalized.

        Returns:
            torch.Tensor: The transformed feature tensor after applying the conditional affine transformation.
        """
        gamma = self.conditional_gamma(context)
        beta = self.conditional_beta(context)

        out = torch.multiply(feature_tensor, gamma + 1)
        out = torch.add(out, beta)
        return self.act(out)


class MainModule(nn.Module):
    def __init__(self, base_module: nn.Module, head_module: nn.Module) -> None:
        """
        This module class holds the main components for prediction in the GPFL model.
        This is mainly used to enable defining one optimizer for the base and head modules.

        Args:
            base_module (nn.Module): Base feature extractor module that generates a feature tensor from the input.
            head_module (nn.Module): Head module that takes a personalized feature tensor and produces the
                final predictions.
        """
        super(MainModule, self).__init__()
        self.base_module = base_module
        self.head_module = head_module


class GpflModel(PartialLayerExchangeModel):
    def __init__(
        self,
        base_module: nn.Module,
        head_module: nn.Module,
        feature_dim: int,
        num_classes: int,
        apply_flatten_features: bool = False,
    ) -> None:
        """
        GPFL model base as described in the paper "GPFL: Simultaneously Learning Global and Personalized
        Feature Information for Personalized Federated Learning." This base module consists of three main
        sub-modules: the main_module, which consists of a feature extractor and a head module; the GCE
        (Global Conditional Embedding) module; and the CoV (Conditional Variance) module.

        Args:
            base_module (nn.Module): Base feature extractor module that generates a feature tensor from the input.
            head_module (nn.Module): Head module that takes a personalized feature tensor and produces the
                final predictions.
            feature_dim (int): The output dimension of the base feature extractor. This is also the input dimension
                of the head and CoV modules.
            num_classes (int): This is used to construct the GCE module.
            apply_flatten_features (bool, optional): Whether the ``base_module``'s output features should be
                flattened or not. Defaults to False.
        """
        super(GpflModel, self).__init__()

        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.main_module = MainModule(base_module, head_module)
        self.gce = GCE(feature_dim, num_classes)
        self.cov = CoV(feature_dim)
        self.apply_flatten_features: bool = apply_flatten_features
        # These modules are exchanged between the server and clients.
        self.modules_to_exchange = ["main_module.base_module", "gce", "cov"]

    def forward(
        self,
        input: torch.Tensor,
        generic_conditional_input: torch.Tensor,
        personalized_conditional_input: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """
        There are two types of forward passes in this model base. The first is the forward pass preformed
        during training. During training, input is passed through the base feature extractor, and then the
        CoV module maps the features into two feature tensors corresponding to local and global features.
        The CoV module requires ``generic_conditional_input`` and ``personalized_conditional_input`` tensors
        , which are used to condition the output of the CoV module. These tensors are computed in clients at the
        beginning of each round. ``local_features`` are fed into the ``head_module`` to produce the final predictions.
        The ``generic_conditional_input`` is used to compute the global features, and these ``global_features``
        are returned only during training.
        The second type of forward pass happens during evaluation. For evaluation, input is passed through the
        base feature extractor, and ``local_features`` generated by the CoV module are used to produce the final
        predictions.

        Args:
            input (torch.Tensor): Input tensor to be fed into the feature extractor.
            generic_conditional_input (torch.Tensor): The conditional input tensor used by the CoV module
                to generate the global features.
            personalized_conditional_input (torch.Tensor): The conditional input tensor used by the CoV module
                to generate the local features.

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]: A tuple in which the first element
                contains a dictionary of predictions and the second element contains intermediate features
                indexed by name.
        """
        features = self.main_module.base_module.forward(input)
        features = self.flatten_features(features) if self.apply_flatten_features else features
        local_features = self.cov(features, personalized_conditional_input)
        predictions = self.main_module.head_module.forward(local_features)
        if not self.training:
            return {"prediction": predictions}, {}

        global_features = self.cov(features, generic_conditional_input)
        return {"prediction": predictions}, {"local_features": local_features, "global_features": global_features}

    def flatten_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        The features tensor is flattened to be of shape are flattened to be of shape (``batch_size``, -1). It is
        expected that the feature tensor is **BATCH FIRST**.

        Args:
            features (torch.Tensor): Features tensor to be flattened. It is assumed that this tensor is
                **BATCH FIRST.**

        Returns:
            torch.Tensor: Flattened feature tensor of shape (``batch_size``, -1)
        """
        return features.reshape(len(features), -1)

    def should_layer_be_exchanged(self, layer_name: str) -> bool:
        """
        Returns True if the "layer_name" corresponds to a layer that exists in any of the modules
        intended to be exchanged.

        Args:
            layer_name (str): String representing the name of the layer in the model's state dictionary.

        Returns:
            bool: True if the layer should be exchanged, False otherwise.
        """
        return any(layer_name.startswith(module_to_exchange) for module_to_exchange in self.modules_to_exchange)

    def layers_to_exchange(self) -> list[str]:
        """
        Returns a list of layer names that should be exchanged between the server and clients.

        Returns:
            list[str]: A list of layer names that should be exchanged. This is used by the
            ``FixedLayerExchanger`` class to determine which layers to exchange during the FL process.
        """
        return [layer_name for layer_name in self.state_dict() if self.should_layer_be_exchanged(layer_name)]
