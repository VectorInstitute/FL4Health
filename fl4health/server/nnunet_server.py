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

    def load_server_model(self, config: Config) -> None:
        plans = pickle.loads(narrow_config_type(config, "nnunet_plans", bytes))
        plans_manager = PlansManager(plans)
        configuration_manager = plans_manager.get_configuration(config["nnunet_config"])
        model = nnUNetTrainer.build_network_architecture(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            int(config["num_input_channels"]),
            int(config["num_segmentation_heads"]),
            bool(config["enable_deep_supervision"]),
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

        # NnUNetClient has serialized nnunet_plans as a property
        plans_bytes = properties["nnunet_plans"]

        # Load server model with plan provided from client
        properties["nnunet_config"] = config["nnunet_config"]
        self.load_server_model(properties)

        # Wrap config functions so that nnunet_plans is included
        new_fit_cfg_fn = add_items_to_config_fn(self.strategy.configure_fit, {"nnunet_plans": plans_bytes})
        new_eval_cfg_fn = add_items_to_config_fn(self.strategy.configure_evaluate, {"nnunet_plans": plans_bytes})
        setattr(self.strategy, "configure_fit", new_fit_cfg_fn)
        setattr(self.strategy, "configure_evaluate", new_eval_cfg_fn)

        # Finish
        self.initialized = True
        log(INFO, "")
