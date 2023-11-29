import os
from logging import DEBUG, INFO
from typing import Optional

import torch
import torch.nn as nn
from flwr.common.logger import log


class WarmUpModel(nn.Module):
    def __init__(self, warm_up: bool = False, warmed_up_dir: Optional[str] = None) -> None:
        super().__init__()
        self.warm_up = warm_up
        if warmed_up_dir:
            log(INFO, "Loading warmed up model weights")
            self.warmed_up_path = os.path.join(warmed_up_dir, "warmed_up_model.pkl")
            self.maybe_init_model()

    def maybe_init_model(self) -> None:
        if not os.path.exists(self.warmed_up_path):
            log(DEBUG, f"Warmed up model not found at {self.warmed_up_path}. Please first run the warm-up script.")
        self = torch.load(self.warmed_up_path)
