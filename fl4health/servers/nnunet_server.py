import pickle
import warnings
from collections.abc import Callable, Sequence
from logging import INFO
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type, Union

import torch.nn as nn
from flwr.common import Parameters
from flwr.common.logger import log
from flwr.common.typing import Code, Config, EvaluateIns, FitIns, GetPropertiesIns, Scalar
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.history import History
from flwr.server.strategy import Strategy

from fl4health.checkpointing.checkpointer import TorchModuleCheckpointer
from fl4health.reporting.base_reporter import BaseReporter
from fl4health.reporting.reports_manager import ReportsManager
from fl4health.servers.base_server import ExchangerType, FlServer
from fl4health.utils.config import narrow_dict_type, narrow_dict_type_and_set_attribute
from fl4health.utils.nnunet_utils import NnunetConfig
from fl4health.utils.parameter_extraction import get_all_model_parameters

with warnings.catch_warnings():
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

FIT_CFG_FN = Callable[[int, Parameters, ClientManager], list[Tuple[ClientProxy, FitIns]]]
EVAL_CFG_FN = Callable[[int, Parameters, ClientManager], list[Tuple[ClientProxy, EvaluateIns]]]
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


class NnunetServer(FlServer):
    def __init__(
        self,
        client_manager: ClientManager,
        fl_config: Config,
        on_init_parameters_config_fn: Callable[[int], Dict[str, Scalar]],
        model: nn.Module | None = None,
        strategy: Strategy | None = None,
        checkpointer: TorchModuleCheckpointer | Sequence[TorchModuleCheckpointer] | None = None,
        reporters: Sequence[BaseReporter] | None = None,
        parameter_exchanger: ExchangerType | None = None,
        intermediate_server_state_dir: Path | None = None,
        server_name: str | None = None,
        accept_failures: bool = True,
        nnunet_trainer_class: Type[nnUNetTrainer] = nnUNetTrainer,
    ) -> None:
        """
        A Basic FlServer with added functionality to ask a client to initialize the global nnunet plans if one was not
        provided in the config. Intended for use with NnUNetClient.

        Args:
            client_manager (ClientManager): Determines the mechanism by which clients are sampled by the server, if
                they are to be sampled at all.
            fl_config (Config): This should be the configuration that was used to setup the federated training.
                In most cases it should be the "source of truth" for how FL training/evaluation should proceed. For
                example, the config used to produce the on_fit_config_fn and on_evaluate_config_fn for the strategy.
                NOTE: This config is DISTINCT from the Flwr server config, which is extremely minimal.
            model (nn.Module): This is the torch model to be hydrated by the _hydrate_model_for_checkpointing function
            on_init_parameters_config_fn (Callable[[int], Dict[str, Scalar]]]): Function used to configure how one
                asks a client to provide parameters from which to initialize all other clients by providing a
                Config dictionary. For NnunetServers this is a required function to provide the additional information
                necessary to a client for parameter initialization
            strategy (Optional[Strategy], optional): The aggregation strategy to be used by the server to handle
                client updates and other information potentially sent by the participating clients. If None the
                strategy is FedAvg as set by the flwr Server.
            checkpointer (TorchCheckpointer | Sequence[TorchCheckpointer], optional): To be provided if the server
                should perform server side checkpointing based on some criteria. If none, then no server-side
                checkpointing is performed. Multiple checkpointers can also be passed in a sequence to checkpoint
                based on multiple criteria. Defaults to None.
            reporters (Sequence[BaseReporter], optional): A sequence of FL4Health reporters which the client should
                send data to.
            intermediate_server_state_dir (Path): A directory to store and load checkpoints from for the server
                during an FL experiment.
            parameter_exchanger (Optional[ExchangerType], optional): A parameter exchanger used to facilitate
                server-side model checkpointing if a checkpointer has been defined. If not provided then checkpointing
                will not be done unless the _hydrate_model_for_checkpointing function is overridden. Because the
                server only sees numpy arrays, the parameter exchanger is used to insert the numpy arrays into a
                provided model. Defaults to None.
            server_name (Optional[str]): An optional string name to uniquely identify server.
            accept_failures (bool, optional): Determines whether the server should accept failures during training or
                evaluation from clients or not. If set to False, this will cause the server to shutdown all clients
                and throw an exception. Defaults to True.
            nnunet_trainer_class (Type[nnUNetTrainer]): nnUNetTrainer class.
                Useful for passing custom nnUNetTrainer. Defaults to the standard nnUNetTrainer class.
                Must match the nnunet_trainer_class passed to the NnunetClient.
        """
        FlServer.__init__(
            self,
            client_manager=client_manager,
            fl_config=fl_config,
            strategy=strategy,
            reporters=reporters,
            model=model,
            checkpointer=checkpointer,
            parameter_exchanger=parameter_exchanger,
            intermediate_server_state_dir=intermediate_server_state_dir,
            on_init_parameters_config_fn=on_init_parameters_config_fn,
            server_name=server_name,
            accept_failures=accept_failures,
        )
        self.nnunet_trainer_class = nnunet_trainer_class

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
        model = self.nnunet_trainer_class.build_network_architecture(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            self.num_input_channels,
            self.num_segmentation_heads,
            self.enable_deep_supervision,
        )

        self.server_model = model

    def update_before_fit(self, num_rounds: int, timeout: Optional[float]) -> None:
        """
        Hook method to allow the server to do some additional initialization prior to fitting. NunetServer
        uses this method to sample a client for properties which are required to initialize the server.

        In particular, if a nnunet_plans file is not provided in the config, this method will sample a client
        which passes the nnunet_plans back to the sever through get_properties RPC. The server then distributes
        the nnunet_plans to the other clients by including it in the config for subsequent FL rounds.

        Even if the nnunet_plans are included in the config, the server will still poll a client in order to have the
        required properties to instantiate the model architecture on the server side which is required for
        checkpointing. These properties include num_segmentation_heads, num_input_channels and enable_deep_supervision.

        Args:
            num_rounds (int): The number of server rounds of FL to be performed
            timeout (Optional[float], optional): The server's timeout parameter. Useful if one is requesting
                information from a client. Defaults to None, which indicates indefinite timeout.
        """

        # If no prior checkpoints exist, initialize server by sampling clients to get required properties to set
        # NOTE: Inherent assumption that if checkpoint exists for server that it also will exist for client.
        if self.per_round_checkpointer is None or not self.per_round_checkpointer.checkpoint_exists():
            # Sample properties from a random client to initialize plans
            log(INFO, "")
            log(INFO, "[PRE-INIT]")
            log(
                INFO,
                "Requesting initialization of global nnunet plans from one random client via get_properties",
            )
            random_client = self._client_manager.sample(1)[0]
            ins = GetPropertiesIns(config=self.fl_config | {"current_server_round": 0})
            properties_res = random_client.get_properties(ins=ins, timeout=timeout, group_id=0)

            if properties_res.status.code == Code.OK:
                log(INFO, "Received global nnunet plans from one random client")
            else:
                raise Exception("Failed to receive properties from client to initialize nnunet plans")

            properties = properties_res.properties

            # Set attributes of server that are dependent on client properties.

            # If config contains nnunet_plans, server side initialization of plans
            # Else client side initialization with nnunet_plans from client
            if self.fl_config.get("nnunet_plans") is not None:
                self.nnunet_plans_bytes = narrow_dict_type(self.fl_config, "nnunet_plans", bytes)
            else:
                self.nnunet_plans_bytes = narrow_dict_type(properties, "nnunet_plans", bytes)
            self.num_segmentation_heads = narrow_dict_type(properties, "num_segmentation_heads", int)
            self.num_input_channels = narrow_dict_type(properties, "num_input_channels", int)
            self.enable_deep_supervision = narrow_dict_type(properties, "enable_deep_supervision", bool)

            self.nnunet_config = NnunetConfig(self.fl_config["nnunet_config"])

            self.initialize_server_model()
        else:
            # If a checkpoint exists, we load in previously checkpointed values for required properties
            self._load_server_state()

        # Wrap config functions so that nnunet_plans is included
        new_fit_cfg_fn = add_items_to_config_fn(self.strategy.configure_fit, {"nnunet_plans": self.nnunet_plans_bytes})
        new_eval_cfg_fn = add_items_to_config_fn(
            self.strategy.configure_evaluate, {"nnunet_plans": self.nnunet_plans_bytes}
        )
        setattr(self.strategy, "configure_fit", new_fit_cfg_fn)
        setattr(self.strategy, "configure_evaluate", new_eval_cfg_fn)

        # Finish
        log(INFO, "")

    # TODO: We should have a get server state method
    # subclass could call parent method and not have to copy entire state.
    def _save_server_state(self) -> None:
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
            "reports_manager": self.reports_manager,
            "server_name": self.server_name,
            "nnunet_plans_bytes": self.nnunet_plans_bytes,
            "num_input_channels": self.num_input_channels,
            "num_segmentation_heads": self.num_segmentation_heads,
            "enable_deep_supervision": self.enable_deep_supervision,
            "nnunet_config": self.nnunet_config,
        }

        self.per_round_checkpointer.save_checkpoint(ckpt)

        log(
            INFO,
            f"Saving server state to checkpoint at {self.per_round_checkpointer.checkpoint_path}",
        )

    def _load_server_state(self) -> None:
        """
        Load server checkpoint consisting of model, history, server name, current round and metrics reporter.
            The method overrides parent to add any necessary state when loading the checkpoint.
        """
        assert self.per_round_checkpointer is not None and self.per_round_checkpointer.checkpoint_exists()

        ckpt = self.per_round_checkpointer.load_checkpoint()

        log(
            INFO,
            f"Loading server state from checkpoint at {self.per_round_checkpointer.checkpoint_path}",
        )

        # Standard attributes to load
        narrow_dict_type_and_set_attribute(self, ckpt, "current_round", "current_round", int)
        narrow_dict_type_and_set_attribute(self, ckpt, "server_name", "server_name", str)
        narrow_dict_type_and_set_attribute(self, ckpt, "reports_manager", "reports_manager", ReportsManager)
        narrow_dict_type_and_set_attribute(self, ckpt, "history", "history", History)
        narrow_dict_type_and_set_attribute(self, ckpt, "model", "parameters", nn.Module, func=get_all_model_parameters)
        # Needed for when _hydrate_model_for_checkpointing is called
        narrow_dict_type_and_set_attribute(self, ckpt, "model", "server_model", nn.Module)

        # NnunetServer specific attributes to load
        narrow_dict_type_and_set_attribute(self, ckpt, "nnunet_plans_bytes", "nnunet_plans_bytes", bytes)
        narrow_dict_type_and_set_attribute(self, ckpt, "num_segmentation_heads", "num_segmentation_heads", int)
        narrow_dict_type_and_set_attribute(self, ckpt, "num_input_channels", "num_input_channels", int)
        narrow_dict_type_and_set_attribute(self, ckpt, "enable_deep_supervision", "enable_deep_supervision", bool)
        narrow_dict_type_and_set_attribute(self, ckpt, "nnunet_config", "nnunet_config", NnunetConfig)
