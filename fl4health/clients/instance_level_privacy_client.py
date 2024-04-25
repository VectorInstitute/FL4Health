from logging import WARNING, INFO
from pathlib import Path
from typing import Optional, Sequence
import itertools

import torch
from flwr.common.logger import log
from flwr.common.typing import Config
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

from fl4health.checkpointing.checkpointer import TorchCheckpointer
from fl4health.clients.basic_client import BasicClient
from fl4health.utils.losses import LossMeterType
from fl4health.utils.metrics import Metric, MetricMeterType


class InstanceLevelPrivacyClient(BasicClient):
    """
    Client for Instance Differentially Private Federated Averaging
    """

    def __init__(
        self,
        data_path: Path,
        metrics: Sequence[Metric],
        device: torch.device,
        loss_meter_type: LossMeterType = LossMeterType.AVERAGE,
        metric_meter_type: MetricMeterType = MetricMeterType.AVERAGE,
        checkpointer: Optional[TorchCheckpointer] = None,
    ) -> None:
        super().__init__(
            data_path=data_path,
            metrics=metrics,
            device=device,
            loss_meter_type=loss_meter_type,
            metric_meter_type=metric_meter_type,
            checkpointer=checkpointer,
        )
        self.clipping_bound: float
        self.noise_multiplier: float

    def setup_client(self, config: Config) -> None:
        # Ensure that clipping bound and noise multiplier is present in config
        # Set attributes to be used when setting DP training
        self.clipping_bound = 5 #self.narrow_config_type(config, "clipping_bound", float)
        self.noise_multiplier = 1 #self.narrow_config_type(config, "noise_multiplier", float)

        # Do basic client setup
        super().setup_client(config)

        # Configure DP training
        self.setup_opacus_objects(config)

    def setup_opacus_objects(self, config: Config) -> None:
        # Validate that the model layers are compatible with privacy mechanisms in Opacus and try to replace the layers
        # with compatible ones if necessary.
        # errors = ModuleValidator.validate(self.model, strict=False)
        # if len(errors) != 0:
        #     for error in errors:
        #         log(WARNING, f"Opacus error: {error}")
        #     self.model = ModuleValidator.fix(self.model)
        
        self.model = self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = self.get_optimizer(config)

        # Create DP training objects
        privacy_engine = PrivacyEngine()
        # NOTE: that Opacus make private is NOT idempotent
        # log(INFO, '<<<<< model start <<<<<')
        # log(INFO, [self.model.state_dict().keys()])
        # log(INFO, '>>>> model end >>>>>')
        # it = itertools.chain.from_iterable(
        #     [param_group["params"] for param_group in self.optimizer.param_groups])
        # log(INFO, '<<<<< optimizer start <<<<<')
        # log(INFO, list(it))
        # log(INFO, '>>>> optimiaer end >>>>>')
        self.noise_multiplier = 1e-16
        self.clipping_bound = 1e16
        
        self.model, self.optimizer, self.train_loader = privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.train_loader,
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.clipping_bound,
            clipping="flat",
            poisson_sampling=False
            # grad_sample_mode="ew"
        )
        # log(INFO, 'finished setting opacus')
        # exit()
