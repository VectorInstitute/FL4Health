import json
import os
from logging import INFO, WARNING
from pathlib import Path
from typing import Optional

import torch
from flwr.common.logger import log


class WarmedUpModule:
    """This class is used to load a pretrained model into the target model."""

    def __init__(
        self,
        pretrained_model: Optional[torch.nn.Module] = None,
        pretrained_model_path: Optional[Path] = None,
        weights_mapping_path: Optional[Path] = None,
    ) -> None:
        """Initialize the WarmedUpModule with the pretrained model states and weights mapping dict.

        Args:
            pretrained_model (Optional[torch.nn.Module]): Pretrained model.
                                                          This is mutually exclusive with pretrained_model_path.
            pretrained_model_path (Optional[Path]): Path of the pretrained model.
                                                    This is mutually exclusive with pretrained_model.
            weights_mapping_dir (Optional[str], optional): Path of to json file of the weights mapping dict.
            If models are not exactly the same, a weights mapping dict is needed to map the weights of the pretrained
            model to the target model.
        """
        if pretrained_model is not None and pretrained_model_path is not None:
            AssertionError(
                "pretrained_model_path and pretrained_model is mutually exclusive. Please provide one of them."
            )

        elif pretrained_model is not None:
            log(INFO, "Pretrained model is provided.")
            self.pretrained_model_state = pretrained_model.state_dict()

        elif pretrained_model_path is not None:
            assert os.path.exists(
                pretrained_model_path
            ), f"Pretrained model path {pretrained_model_path} does not exist."
            log(INFO, f"Loading pretrained model from {pretrained_model_path}")
            self.pretrained_model_state = torch.load(pretrained_model_path).state_dict()

        else:
            raise AssertionError("At least one of pretrained_model_path and pretrained_model should be provided.")

        if weights_mapping_path is not None:
            with open(weights_mapping_path, "r") as file:
                self.weights_mapping_dict = json.load(file)
        else:
            log(INFO, "Weights mapping dict is not provided. Matching states directly, based on target model's keys.")
            self.weights_mapping_dict = None

    def get_matching_component(self, key: str) -> Optional[str]:
        """Get the matching component of the key from the weights mapping dictionary. Since the provided mapping
        can contain partial names of the keys, this function is used to split the key of the target model and
        match it with the partial key in the mapping, returning the complete name of the key in the pretrained model.

        This allows users to provide one mapping for multiple states that share the same prefix. For example,if the
        mapping is {"model": "global_model"} and the input key of the target model is "model.layer1.weight",then the
        returned matching component is "global_model.layer1.weight".

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
                matching_component = component
            else:
                matching_component = f"{matching_component}.{component}"
            if matching_component in self.weights_mapping_dict:
                return self.weights_mapping_dict[matching_component] + key[len(matching_component) :]
        return None

    def load_from_pretrained(self, model: torch.nn.Module) -> torch.nn.Module:
        """Load the pretrained model into the target model.

        Args:
            model (torch.nn.Module): target model.
        """

        assert self.pretrained_model_state is not None

        target_model_state = model.state_dict()

        matching_state = {}
        for key in target_model_state.keys():
            original_state = target_model_state[key]

            pretrained_key = self.get_matching_component(key)
            log(INFO, f"Matching: {key} -> {pretrained_key}")
            if pretrained_key is not None:
                if pretrained_key in self.pretrained_model_state.keys():
                    pretrained_state = self.pretrained_model_state[pretrained_key]
                    if original_state.size() == pretrained_state.size():
                        matching_state[key] = pretrained_state
                        log(INFO, "Successful matching states.")
                    else:
                        log(
                            WARNING,
                            f"State won't be loaded. Mismatched sizes {original_state.size()}) -> ({pretrained_key}).",
                        )
                else:
                    log(
                        WARNING,
                        f"state won't be loaded. Key {pretrained_key} not found in the pretrained model states.",
                    )

        log(INFO, f"{len(matching_state)}/{len(target_model_state)} states got matched.")

        target_model_state.update(matching_state)
        model.load_state_dict(target_model_state)
        return model
