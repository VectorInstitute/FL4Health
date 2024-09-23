import pickle
import warnings
from logging import INFO
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import torch.nn as nn
from flwr.common import Parameters
from flwr.common.logger import log
from flwr.common.typing import Code, Config, EvaluateIns, FitIns, GetPropertiesIns
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.history import History
from flwr.server.strategy import Strategy

from fl4health.checkpointing.checkpointer import TorchCheckpointer
from fl4health.parameter_exchange.parameter_exchanger_base import ParameterExchanger
from fl4health.reporting.fl_wandb import ServerWandBReporter
from fl4health.reporting.metrics import MetricsReporter
from fl4health.server.base_server import FlServerWithCheckpointing, FlServerWithInitializer
from fl4health.utils.config import narrow_config_type
from fl4health.utils.nnunet_utils import NnunetConfig
from fl4health.utils.parameter_extraction import get_all_model_parameters

with warnings.catch_warnings():
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

FIT_CFG_FN = Callable[[int, Parameters, ClientManager], List[Tuple[ClientProxy, FitIns]]]
EVAL_CFG_FN = Callable[[int, Parameters, ClientManager], List[Tuple[ClientProxy, EvaluateIns]]]
CFG_FN = Union[FIT_CFG_FN, EVAL_CFG_FN]


def add_items_to_config_fn(fn: CFG_FN, items: Config) -> CFG_FN:
    """
    Accepts a flwr Strategy configure function (either configure_fit or
    configure_evaluate) and returns a new function that returns the same thing
    except the dictionary items in the items argument have been added to the
    config that is returned by the original function

    Args:
        fn (CFG_FN): The Strategy configure function to wrap
        items (Config): A Config containing additional items to update the
            original config with

    Returns:
        CFG_FN: The wrapped function. Argument and return type is the same
    """

    def new_fn(*args: Any, **kwargs: Any) -> Any:
        cfg_ins = fn(*args, **kwargs)
        for _, ins in cfg_ins:
            ins.config.update(items)
        return cfg_ins

    return new_fn


class NnunetServer(FlServerWithInitializer, FlServerWithCheckpointing):
    def __init__(
        self,
        client_manager: ClientManager,
        parameter_exchanger: ParameterExchanger,
        model: Optional[nn.Module] = None,
        wandb_reporter: Optional[ServerWandBReporter] = None,
        strategy: Optional[Strategy] = None,
        checkpointer: Optional[Union[TorchCheckpointer, Sequence[TorchCheckpointer]]] = None,
        metrics_reporter: Optional[MetricsReporter] = None,
        intermediate_server_state_dir: Optional[Path] = None,
        server_name: Optional[str] = None,
    ) -> None:
        FlServerWithCheckpointing.__init__(
            self,
            client_manager=client_manager,
            model=model,
            parameter_exchanger=parameter_exchanger,
            wandb_reporter=wandb_reporter,
            strategy=strategy,
            checkpointer=checkpointer,
            metrics_reporter=metrics_reporter,
            intermediate_server_state_dir=intermediate_server_state_dir,
            server_name=server_name,
        )
        """
        A Basic FlServer with added functionality to ask a client to initialize
        the global nnunet plans if one was not provided in the config. Intended
        for use with NnUNetClient
        """
        self.initialized = False

        self.nnunet_plans_bytes: bytes
        self.num_input_channels: int
        self.num_segmentation_heads: int
        self.enable_deep_supervision: bool
        self.nnunet_config: NnunetConfig

    def initialize_server_model(self) -> None:
        # Ensure required attributes are set
        assert (
            self.nnunet_plans_bytes is not None
            and self.num_input_channels is not None
            and self.num_segmentation_heads is not None
            and self.enable_deep_supervision is not None
            and self.nnunet_config is not None
        )

        plans = pickle.loads(self.nnunet_plans_bytes)
        plans_manager = PlansManager(plans)
        configuration_manager = plans_manager.get_configuration(self.nnunet_config.value)
        model = nnUNetTrainer.build_network_architecture(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            self.num_input_channels,
            self.num_segmentation_heads,
            self.enable_deep_supervision,
        )

        self.server_model = model

    def fit(self, num_rounds: int, timeout: Optional[float]) -> Tuple[History, float]:
        """
        Same as parent method except initialize hook method is called first
        """
        # Initialize the server
        if not self.initialized:
            self.initialize(server_round=0, timeout=timeout)

        return FlServerWithCheckpointing.fit(self, num_rounds, timeout)

    def initialize(self, server_round: int, timeout: Optional[float] = None) -> None:
        # Get fit config
        dummy_params = Parameters([], "None")
        config = self.strategy.configure_fit(server_round, dummy_params, self._client_manager)[0][1].config

        # Check if plans need to be initialized
        if config.get("nnunet_plans") is not None:
            self.initialized = True
            return

        # If no prior checkpoints exist, initialize server by sampling clients to get required properties to set
        if self.per_round_checkpointer is None or not self.per_round_checkpointer.checkpoint_exists():
            # Sample properties from a random client to initialize plans
            log(INFO, "")
            log(INFO, "[PRE-INIT]")
            log(INFO, "Requesting initialization of global nnunet plans from one random client via get_properties")
            random_client = self._client_manager.sample(1)[0]
            ins = GetPropertiesIns(config=config)
            properties_res = random_client.get_properties(ins=ins, timeout=timeout, group_id=server_round)

            if properties_res.status.code == Code.OK:
                log(INFO, "Recieved global nnunet plans from one random client")
            else:
                raise Exception("Failed to receive properties from client to initialize nnunet plans")

            properties = properties_res.properties

            # Set attributes of server that are dependent on client properties.
            # NnUNetClient has serialized nnunet_plans as a property
            self.nnunet_plans_bytes = narrow_config_type(properties, "nnunet_plans", bytes)
            self.num_segmentation_heads = narrow_config_type(properties, "num_segmentation_heads", int)
            self.num_input_channels = narrow_config_type(properties, "num_input_channels", int)
            self.enable_deep_supervision = narrow_config_type(properties, "enable_deep_supervision", bool)

            self.nnunet_config = NnunetConfig(config["nnunet_config"])

            self.initialize_server_model()
        else:
            # If a checkpoint exists, we load in previously checkpointed values for required properties
            self.load_server_state()

        # Wrap config functions so that nnunet_plans is included
        new_fit_cfg_fn = add_items_to_config_fn(self.strategy.configure_fit, {"nnunet_plans": self.nnunet_plans_bytes})
        new_eval_cfg_fn = add_items_to_config_fn(
            self.strategy.configure_evaluate, {"nnunet_plans": self.nnunet_plans_bytes}
        )
        setattr(self.strategy, "configure_fit", new_fit_cfg_fn)
        setattr(self.strategy, "configure_evaluate", new_eval_cfg_fn)

        # Finish
        self.initialized = True
        log(INFO, "")

    def save_server_state(self) -> None:
        """
        Save server checkpoint consisting of model, history, server round, metrics reporter and server name.
            This method overrides parent to also checkpoint nnunet_plans, num_input_channels,
            num_segmentation_heads and enable_deep_supervision.
        """

        assert self.per_round_checkpointer is not None

        assert (
            self.nnunet_plans_bytes is not None
            and self.num_input_channels is not None
            and self.num_segmentation_heads is not None
            and self.enable_deep_supervision is not None
            and self.nnunet_config is not None
        )

        ckpt = {
            "model": self.server_model,
            "history": self.history,
            "current_round": self.current_round,
            "metrics_reporter": self.metrics_reporter,
            "server_name": self.server_name,
            "nnunet_plans_bytes": self.nnunet_plans_bytes,
            "num_input_channels": self.num_input_channels,
            "num_segmentation_heads": self.num_segmentation_heads,
            "enable_deep_supervision": self.enable_deep_supervision,
            "nnunet_config": self.nnunet_config,
        }

        self.per_round_checkpointer.save_checkpoint(ckpt)

        log(INFO, f"Saving server state to checkpoint at {self.per_round_checkpointer.checkpoint_path}")

    def load_server_state(self) -> None:
        """
        Load server checkpoint consisting of model, history, server name, current round and metrics reporter.
            The method overrides parent to add any necessary state when loading the checkpoint.
        """
        assert self.per_round_checkpointer is not None and self.per_round_checkpointer.checkpoint_exists()

        ckpt = self.per_round_checkpointer.load_checkpoint()

        assert "model" in ckpt and isinstance(ckpt["model"], nn.Module)
        assert "server_name" in ckpt and isinstance(ckpt["server_name"], str)
        assert "current_round" in ckpt and isinstance(ckpt["current_round"], int)
        assert "metrics_reporter" in ckpt and isinstance(ckpt["metrics_reporter"], MetricsReporter)
        assert "history" in ckpt and isinstance(ckpt["history"], History)

        assert "nnunet_plans_bytes" in ckpt and isinstance(ckpt["nnunet_plans_bytes"], bytes)
        assert "num_segmentation_heads" in ckpt and isinstance(ckpt["num_segmentation_heads"], int)
        assert "num_input_channels" in ckpt and isinstance(ckpt["num_input_channels"], int)
        assert "enable_deep_supervision" in ckpt and isinstance(ckpt["enable_deep_supervision"], bool)
        assert "nnunet_config" in ckpt and isinstance(ckpt["nnunet_config"], NnunetConfig)

        log(INFO, f"Loading server state from checkpoint at {self.per_round_checkpointer.checkpoint_path}")

        self.current_round = ckpt["current_round"]
        self.server_name = ckpt["server_name"]
        self.metrics_reporter = ckpt["metrics_reporter"]
        self.history = ckpt["history"]
        self.parameters = get_all_model_parameters(ckpt["model"])

        self.server_model = ckpt["model"]
        self.nnunet_plans_bytes = ckpt["nnunet_plans_bytes"]
        self.num_segmentation_heads = ckpt["num_segmentation_heads"]
        self.num_input_channels = ckpt["num_input_channels"]
        self.enable_deep_supervision = ckpt["enable_deep_supervision"]
        self.nnunet_config = ckpt["nnunet_config"]
