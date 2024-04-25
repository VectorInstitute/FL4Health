from logging import INFO
from typing import Callable, Dict, List

import torch
import torch.nn as nn
from flwr.common.logger import log
from torch.utils.hooks import RemovableHandle


class FeatureExtractorBuffer:
    def __init__(self, model: nn.Module, flatten_feature_extraction_layers: Dict[str, bool]) -> None:
        self.model = model
        self.flatten_feature_extraction_layers = flatten_feature_extraction_layers
        self.fhooks: List[RemovableHandle] = []

        self.accumulate_features: bool = False
        self.extracted_features_buffers: Dict[str, List[torch.Tensor]] = {
            layer: [] for layer in flatten_feature_extraction_layers.keys()
        }

    def enable_accumulating_features(self) -> None:
        # Accumulate features in the buffers for multiple forward passes.
        self.accumulate_features = True

    def disable_accumulating_features(self) -> None:
        # Do not accumulate features in the buffers and overwrite them for each forward pass.
        self.accumulate_features = False

    def clear_buffers(self) -> None:
        # Clear the buffers for the extracted features.
        self.extracted_features_buffers = {layer: [] for layer in self.flatten_feature_extraction_layers.keys()}

    def _maybe_register_hooks(self) -> None:
        # Register hooks if they are not already registered. Hooks extract the output of the selected layers.
        if len(self.fhooks) == 0:
            log(INFO, "Starting to register hooks:")
            named_layers = dict(self.model.named_modules())
            for layer in named_layers.keys():
                if layer in self.flatten_feature_extraction_layers.keys():
                    log(INFO, f"Registering hook for layer: {layer}")
                    self.fhooks.append(getattr(self.model, layer).register_forward_hook(self.forward_hook(layer)))
        else:
            log(INFO, "Hooks already registered.")

    def remove_hooks(self) -> None:
        # We need to remove the hooks prior to checkpointing the model.
        log(INFO, "Removing hooks.")
        for hook in self.fhooks:
            hook.remove()
        self.fhooks.clear()

    def forward_hook(self, layer_name: str) -> Callable:
        def hook(module: nn.Module, input: torch.Tensor, output: torch.Tensor) -> None:
            if not self.accumulate_features:
                self.extracted_features_buffers[layer_name] = [output]
            else:
                self.extracted_features_buffers[layer_name].append(output)

        return hook

    def flatten(self, features: torch.Tensor) -> torch.Tensor:
        # The features are of shape (batch_size, *). We flatten them accross the batch dimension
        # to get a 2D tensor of shape (batch_size, feature_size).
        return features.reshape(len(features), -1)

    def get_extracted_features(self) -> Dict[str, torch.Tensor]:
        features = {}

        for layer in self.extracted_features_buffers:
            # The buffers are in shape (batch_size, feature_size). We tack them along the batch dimension
            # (dim=0) to get a tensor of shape (num_samples, feature_size)
            features[layer] = (
                self.flatten(torch.cat(self.extracted_features_buffers[layer], dim=0))
                if self.flatten_feature_extraction_layers[layer]
                else torch.cat(self.extracted_features_buffers[layer], dim=0)
            )

        return features
