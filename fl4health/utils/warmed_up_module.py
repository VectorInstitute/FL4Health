import json
import os
from logging import ERROR, INFO
from typing import Optional

import torch
from flwr.common.logger import log


class WarmedUpModule:
    """This class is used to load a pretrained model into the current model."""

    def __init__(self, pretrained_model_dir: Optional[str], weights_mapping_dir: Optional[str] = None) -> None:
        """Initialize the WarmedUpModule with the pretrained model stats and weights mapping dict.

        Args:
            pretrained_model_dir (Optional[str]): Directory of the pretrained model
            weights_mapping_dir (Optional[str], optional): Directory of to json file of the weights mapping dict.
            Defaults to None.
            If models are not exactly the same, a weights mapping dict is needed to map the weights of the pretrained
            model to the current model.
        """

        if pretrained_model_dir is None or not os.path.exists(pretrained_model_dir):
            log(INFO, "No pretrained model provided")
            self.pretrained_model_state = None
        else:
            log(INFO, f"Loading pretrained model from {pretrained_model_dir}")
            self.pretrained_model_state = torch.load(pretrained_model_dir).state_dict()

        if weights_mapping_dir is not None:
            with open(weights_mapping_dir, "r") as file:
                self.weights_mapping_dict = json.load(file)
                log(INFO, f"Weights mapping dict: {self.weights_mapping_dict}")
        else:
            self.weights_mapping_dict = None

    def get_matching_component(self, key: str) -> Optional[str]:
        """Get the matching component of the key from the weights mapping dict.

        Args:
            key (str): Key to be matched in pretrained model.

        Returns:
            Optional[str]: If no weights mapping dict is provided, returns the key. Otherwise, if the key is in the
            weights mapping dict, returns the matching component of the key. Otherwise, returns None.
        """

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

    def load_from_pretrained(self, model: torch.nn.Module) -> None:
        """Load the pretrained model into the current model.

        Args:
            model (torch.nn.Module): Current model.
        """

        assert self.pretrained_model_state is not None

        current_model_state = model.state_dict()

        matching_state = {}
        for key in current_model_state.keys():
            pretrained_key = self.get_matching_component(key)
            log(INFO, f"Matching: {key} -> {pretrained_key}")
            original_state = current_model_state[key]
            if pretrained_key in self.pretrained_model_state.keys():
                pretrained_state = self.pretrained_model_state[pretrained_key]
            else:
                continue
            if original_state.size() == pretrained_state.size():
                matching_state[key] = pretrained_state
            else:
                log(ERROR, f"Dismatched sizes {original_state.size()} -> {pretrained_state.size()}")
        log(INFO, f"!!! {len(matching_state)}/{len(current_model_state)} STATS GOT MATCHED !!!")

        current_model_state.update(matching_state)
        model.load_state_dict(current_model_state)
