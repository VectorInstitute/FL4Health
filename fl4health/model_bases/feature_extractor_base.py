from typing import List, Optional, Tuple, Dict, Callable, Any, Sequence
from collections import OrderedDict

import torch
import torch.nn as nn

class FeatureExtractorModel:
    def __init__(self, model: nn.Module, output_layers: Sequence[str], flatten_features: Optional[Sequence[bool]])-> None:
        self.model = model
        self.output_layers = output_layers
        if flatten_features is None:
            self.flatten_features: Sequence[bool] =  [True for _ in range(len(self.feature_extraction_layers))]
        else:
            self.flatten_features = flatten_features
        if len(self.flatten_features) != len(self.output_layers):
            raise ValueError("flatten_features must be the same length as output_layers")
        self.selected_out: Dict = OrderedDict()
        self.fhooks = []
        named_layers = dict(self.model.named_modules())
        for i,layer in enumerate(named_layers.keys()):
            if i in self.output_layers:
                self.fhooks.append(getattr(self.model,layer).register_forward_hook(self.forward_hook(layer)))
    
    def forward_hook(self,layer_name:str) -> Callable:
        def hook(module: nn.Module, input:torch.Tensor, output:torch.Tensor) -> None:
            self.selected_out[layer_name] = output
        return hook
    
    def __getattr__(self, attr: str) -> Any:
        if hasattr(self.model, attr):
            return getattr(self.model, attr)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr}'")
    
    def flatten(self, features: torch.Tensor) -> torch.Tensor:
        # The features are of shape (batch_size, *). We flatten them accross the batch dimension
        # to get a 2D tensor of shape (batch_size, feature_size).
        return features.reshape(len(features), -1)

    
    def forward(self, input: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        output = self.model.forward(input)

        out, features = (output, None) if not isinstance(output, tuple) else output

        features = {} if features is None else features

        for i, layer in enumerate(self.selected_out):
            features[layer] = self.flatten(self.selected_out[layer]) if self.flatten_features[layer] else self.selected_out[layer]

        return out, features
