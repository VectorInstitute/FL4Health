import json
from abc import ABC
from logging import ERROR, INFO
from typing import Any, Dict, Optional

import torch
from flwr.common.logger import log


class WarmedUpModule(ABC):
    def __init__(self, pretrained_model_dir: str, weights_mapping_dir: Optional[str] = None) -> None:

        self.pretrained_model_dir = pretrained_model_dir
        self.weights_mapping_dict: Optional[Dict[str, str]] = None

        if weights_mapping_dir is not None:
            with open(weights_mapping_dir, "r") as file:
                self.weights_mapping_dict = json.load(file)
                log(INFO, f"Loaded weights mapping dict: {self.weights_mapping_dict}")

    def get_matching_component(self, key: str) -> Optional[str]:
        if self.weights_mapping_dict is None:
            return key

        components = key.split(".")

        for i, component in enumerate(components):
            if i == 0:
                matching_component = components[0]
            else:
                matching_component += "." + component
            if matching_component in self.weights_mapping_dict:
                return self.weights_mapping_dict[matching_component] + key[len(matching_component) :]
        return None

    def load_from_pretrained(self, model: Any) -> None:

        assert self.pretrained_model_dir is not None
        log(INFO, f"Loading pretrained model from {self.pretrained_model_dir}")

        current_model_state = model.state_dict()
        pretrained_model_state = torch.load(self.pretrained_model_dir).state_dict()

        matching_state = {}
        for key in current_model_state.keys():
            pretrained_key = self.get_matching_component(key)
            log(INFO, f"Matching: {key} -> {pretrained_key}")
            original_state = current_model_state[key]
            if pretrained_key in pretrained_model_state.keys():
                pretrained_state = pretrained_model_state[pretrained_key]
            else:
                continue
            if original_state.size() == pretrained_state.size():
                matching_state[key] = pretrained_state
            else:
                log(ERROR, f"Dismatched sizes {original_state.size()} -> {pretrained_state.size()}")
        log(INFO, f"!!! {len(matching_state)}/{len(current_model_state)} STATS GOT MATCHED !!!")

        current_model_state.update(matching_state)
        model.load_state_dict(current_model_state)
