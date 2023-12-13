from abc import ABC
from logging import INFO
from typing import Any, Dict, Optional

import torch
from flwr.common.logger import log


class ModelSurgery(ABC):
    def __init__(
        self, pretrained_model_dir: Optional[str] = None, load_weights_map: Optional[Dict[str, str]] = None
    ) -> None:
        self.pretrained_model_dir = pretrained_model_dir

    def _maybe_load_from_pretrained(self, model: Any) -> None:

        current_model_state = model.state_dict()
        if self.pretrained_model_dir is None:
            print(current_model_state.keys())
            return

        pretrained_model_state = torch.load(self.pretrained_model_dir).state_dict()

        matching_state = {}
        for k, v in pretrained_model_state.items():
            if k in current_model_state:
                if v.size() == current_model_state[k].size():
                    matching_state[k] = v
                elif current_model_state[k].size()[1:] == v.size()[1:]:
                    repeat = current_model_state[k].size()[0] // v.size()[0]
                    original_size = tuple([1] * (len(current_model_state[k].size()) - 1))
                    matching_state[k] = v.repeat((repeat,) + original_size)

        log(INFO, f"matching state: {len(matching_state)}")
        current_model_state.update(matching_state)
        model.load_state_dict(current_model_state)
