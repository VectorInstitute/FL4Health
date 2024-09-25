from logging import INFO
from typing import Callable, Dict, List

import torch
import torch.nn as nn
from flwr.common.logger import log
from torch.utils.hooks import RemovableHandle


class FeatureExtractorBuffer:
    def __init__(self, model: nn.Module, flatten_feature_extraction_layers: Dict[str, bool]) -> None:
        """
        This class is used to extract features from the intermediate layers of a neural network model and store them in
        a buffer. The features are extracted using additional hooks that are registered to the model. The extracted
        features are stored in a dictionary where the keys are the layer names and the values are the extracted
        features as torch Tensors.

        Args:
            model (nn.Module): The neural network model.
            flatten_feature_extraction_layers (Dict[str, bool]): Dictionary of layers to extract features from them and
            whether to flatten them. Keys are the layer names that are extracted from the named_modules and values are
            boolean.
        Attributes:
            model (nn.Module): The neural network model.
            flatten_feature_extraction_layers (Dict[str, bool]): A dictionary specifying whether to flatten the feature
                extraction layers.
            fhooks (List[RemovableHandle]): A list to store the handles for removing hooks.
            accumulate_features (bool): A flag indicating whether to accumulate features.
            extracted_features_buffers (Dict[str, List[torch.Tensor]]): A dictionary to store the extracted features
                for each layer.
        """
        self.model = model
        self.flatten_feature_extraction_layers = flatten_feature_extraction_layers
        self.fhooks: List[RemovableHandle] = []

        self.accumulate_features: bool = False
        self.extracted_features_buffers: Dict[str, List[torch.Tensor]] = {
            layer: [] for layer in flatten_feature_extraction_layers.keys()
        }

    def enable_accumulating_features(self) -> None:
        """
        Enables the accumulation of features in the buffers for multiple forward passes.

        This method sets the `accumulate_features` flag to True, allowing the model to accumulate features
        in the buffers for multiple forward passes. This can be useful in scenarios where you want to extract
        features from intermediate layers of the model during inference.
        """
        self.accumulate_features = True

    def disable_accumulating_features(self) -> None:
        """
        Disables the accumulation of features in the buffers.

        This method sets the `accumulate_features` attribute to False, which prevents the buffers from accumulating
        features and overwrites them for each forward pass.
        """
        self.accumulate_features = False

    def clear_buffers(self) -> None:
        """
        Clears the extracted features buffers for all layers.
        """
        self.extracted_features_buffers = {layer: [] for layer in self.flatten_feature_extraction_layers.keys()}

    def get_hierarchical_attr(self, module: nn.Module, layer_hierarchy: List[str]) -> nn.Module:
        """
        Traverse the hierarchical attributes of the module to get the desired attribute. Hooks should be
        registered to specific layers of the model, not to nn.Sequential or nn.ModuleList.

        Args:
            module (nn.Module): The nn.Module object to traverse.
            layer_hierarchy (List[str]): The hierarchical list of name of desired layer.

        Returns:
            nn.Module: The desired layer of the model.
        """
        if len(layer_hierarchy) == 1:
            return getattr(module, layer_hierarchy[0])
        else:
            return self.get_hierarchical_attr(getattr(module, layer_hierarchy[0]), layer_hierarchy[1:])

    def find_last_common_prefix(self, prefix: str, layers_name: List[str]) -> str:
        """
        Check the model's list of named modules to filter any layer that starts with the given prefix and
        return the last one.

        Args:
            prefix (str): The prefix of the layer name for registering the hook.
            layers_name (List[str]): The list of named modules of the model. The assumption is that list of
            named modules is sorted in the order of the model's forward pass with depth-first traversal. This
            will allow the user to specify the generic name of the layer instead of the full hierarchical name.

        Returns:
            str: The complete name of last named layer that matches the prefix.
        """
        filtered_layers = [layer for layer in layers_name if layer.startswith(prefix)]

        # Return the last element that matches the criteria
        return filtered_layers[-1]

    def _maybe_register_hooks(self) -> None:
        """
        Checks if hooks are already registered and registers them if not.
        Hooks extract the intermediate feature as output of the selected layers in the model.
        """
        if len(self.fhooks) == 0:
            log(INFO, "Starting to register hooks:")
            named_layers = list(dict(self.model.named_modules()).keys())
            for layer in self.flatten_feature_extraction_layers.keys():
                log(INFO, f"Registering hook for layer: {layer}")
                # Find the last specific layer under a given generic name
                specific_layer = self.find_last_common_prefix(layer, named_layers)
                # Split the specific layer name by '.' to get the hierarchical attribute
                layer_hierarchicy_list = specific_layer.split(".")
                self.fhooks.append(
                    self.get_hierarchical_attr(self.model, layer_hierarchicy_list).register_forward_hook(
                        self.forward_hook(layer)
                    )
                )
        else:
            log(INFO, "Hooks already registered.")

    def remove_hooks(self) -> None:
        """
        Removes the hooks from the model for checkpointing and clears the hook list. This method is used to remove
        any hooks that have been added to the feature extractor buffer. It is typically called prior to checkpointing
        the model.
        """

        log(INFO, "Removing hooks.")
        for hook in self.fhooks:
            hook.remove()
        self.fhooks.clear()

    def forward_hook(self, layer_name: str) -> Callable:
        """
        Returns a hook function that is called during the forward pass of a module.

        Args:
            layer_name (str): The name of the layer.

        Returns:
            Callable: The hook function that takes in a module, input, and output tensors.
        """

        def hook(module: nn.Module, input: torch.Tensor, output: torch.Tensor) -> None:
            if not self.accumulate_features:
                self.extracted_features_buffers[layer_name] = [output]
            else:
                self.extracted_features_buffers[layer_name].append(output)

        return hook

    def flatten(self, features: torch.Tensor) -> torch.Tensor:
        """
        Flattens the input tensor along the batch dimension. The features are of shape (batch_size, *).
        We flatten them across the batch dimension to get a 2D tensor of shape (batch_size, feature_size).

        Args:
            features (torch.Tensor): The input tensor of shape (batch_size, *).

        Returns:
            torch.Tensor: The flattened tensor of shape (batch_size, feature_size).
        """

        return features.reshape(len(features), -1)

    def get_extracted_features(self) -> Dict[str, torch.Tensor]:
        """
        Returns a dictionary of extracted features.

        Returns:
            features (Dict[str, torch.Tensor]): A dictionary where the keys are the layer names and the values are
                the extracted features as torch Tensors.
        """
        features = {}

        for layer in self.extracted_features_buffers:
            features[layer] = (
                self.flatten(torch.cat(self.extracted_features_buffers[layer], dim=0))
                if self.flatten_feature_extraction_layers[layer]
                else torch.cat(self.extracted_features_buffers[layer], dim=0)
            )

        return features
