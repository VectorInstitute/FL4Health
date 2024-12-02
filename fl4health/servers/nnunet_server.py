import pickle
import warnings
from collections.abc import Callable, Sequence
from logging import INFO
from typing import Any, Dict, Optional, Tuple, Type, Union

import torch.nn as nn
from flwr.common import Parameters
from flwr.common.logger import log
from flwr.common.typing import Code, Config, EvaluateIns, FitIns, GetPropertiesIns, Scalar
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.history import History
from flwr.server.strategy import Strategy

from fl4health.checkpointing.server_module import NnUnetServerCheckpointAndStateModule
from fl4health.reporting.base_reporter import BaseReporter
from fl4health.reporting.reports_manager import ReportsManager
from fl4health.servers.base_server import FlServer
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
        strategy: Strategy | None = None,
        reporters: Sequence[BaseReporter] | None = None,
        checkpoint_and_state_module: NnUnetServerCheckpointAndStateModule | None = None,
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
            on_init_parameters_config_fn (Callable[[int], Dict[str, Scalar]]): Function used to configure how one
                asks a client to provide parameters from which to initialize all other clients by providing a
                Config dictionary. For NnunetServers this is a required function to provide the additional information
                necessary to a client for parameter initialization
            strategy (Strategy | None, optional): The aggregation strategy to be used by the server to handle
                client updates and other information potentially sent by the participating clients. If None the
                strategy is FedAvg as set by the flwr Server. Defaults to None.
            reporters (Sequence[BaseReporter] | None, optional): A sequence of FL4Health reporters which the client
                should send data to. Defaults to None.
            checkpoint_and_state_module (NnUnetServerCheckpointAndStateModule | None, optional): This module is used
                to handle both model checkpointing and state checkpointing. The former is aimed at saving model
                artifacts to be used or evaluated after training. The later is used to preserve training state
                (including models) such that if FL training is interrupted, the process may be restarted. If no
                module is provided, no checkpointing or state preservation will happen. Defaults to None.
                NOTE: For NnUnet, this module is allowed to have all components defined other than the model, as it
                may be set later when the server asks the clients to provide the architecture.
            server_name (str | None, optional): An optional string name to uniquely identify server. This name is also
                used as part of any state checkpointing done by the server. Defaults to None.
            accept_failures (bool, optional): Determines whether the server should accept failures during training or
                evaluation from clients or not. If set to False, this will cause the server to shutdown all clients
                and throw an exception. Defaults to True.
            nnunet_trainer_class (Type[nnUNetTrainer]): nnUNetTrainer class.
                Useful for passing custom nnUNetTrainer. Defaults to the standard nnUNetTrainer class.
                Must match the nnunet_trainer_class passed to the NnunetClient.
        """
        if checkpoint_and_state_module is not None:
            assert isinstance(
                checkpoint_and_state_module,
                NnUnetServerCheckpointAndStateModule,
            ), "checkpoint_and_state_module must have type NnUnetServerCheckpointAndStateModule"
        super().__init__(
            client_manager=client_manager,
            fl_config=fl_config,
            strategy=strategy,
            reporters=reporters,
            checkpoint_and_state_module=checkpoint_and_state_module,
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

        self.checkpoint_and_state_module.model = model

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

        server_nnunet_plans_exist = self.fl_config.get("nnunet_plans") is not None
        state_checkpointer_exists = self.checkpoint_and_state_module.state_checkpointer is not None

        # If the state_checkpointer has been specified and a state checkpoint exists, we load state
        # NOTE: Inherent assumption that if checkpoint exists for server that it also will exist for client.
        if (
            self.checkpoint_and_state_module.state_checkpointer is not None
            and self.checkpoint_and_state_module.state_checkpointer.checkpoint_exists(self.state_checkpoint_name)
        ):
            self._load_server_state()
        # Otherwise, we're starting training from "scratch"
        elif state_checkpointer_exists or not server_nnunet_plans_exist:
            # 1) If the state checkpointer is not None, then we want to do state checkpointing. So we need information
            #       from the clients in the form of get_properties.
            # 2) If the nnUnet plans are not specified, we also need those plans from the client.
            # In either case, we query clients for the information
            log(INFO, "")
            log(INFO, "[PRE-INIT]")
            log(INFO, "Requesting properties from one random client via get_properties")

            if not server_nnunet_plans_exist:
                log(INFO, "Initialization of global nnunet plans will be sourced from this client")
            if state_checkpointer_exists:
                log(
                    INFO,
                    "Properties from NnUnetTrainer will be sourced from this client to facilitate state preservation",
                )

            random_client = self._client_manager.sample(1)[0]
            ins = GetPropertiesIns(config=self.fl_config | {"current_server_round": 0})
            properties_res = random_client.get_properties(ins=ins, timeout=timeout, group_id=0)

            if properties_res.status.code == Code.OK:
                log(INFO, "Received properties from one random client")
            else:
                raise Exception("Failed to successfully receive properties from client")
            properties = properties_res.properties

            # Set attributes of server that are dependent on client properties.

            # If config contains nnunet_plans, server side initialization of plans
            # Else client side initialization with nnunet_plans from client
            if server_nnunet_plans_exist:
                self.nnunet_plans_bytes = narrow_dict_type(self.fl_config, "nnunet_plans", bytes)
            else:
                self.nnunet_plans_bytes = narrow_dict_type(properties, "nnunet_plans", bytes)

            self.num_segmentation_heads = narrow_dict_type(properties, "num_segmentation_heads", int)
            self.num_input_channels = narrow_dict_type(properties, "num_input_channels", int)
            self.enable_deep_supervision = narrow_dict_type(properties, "enable_deep_supervision", bool)

        if self.per_round_checkpointer is None or not self.per_round_checkpointer.checkpoint_exists():
            # If we're starting training from scratch, set the nnunet_config property and initialize the server model
            self.nnunet_config = NnunetConfig(self.fl_config["nnunet_config"])
            self.initialize_server_model()

        # Wrap config functions so that we are sure the nnunet_plans are included
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
        Save server checkpoint consisting of model, history, server round, metrics reporter and server name. This
        method overrides parent to also checkpoint nnunet_plans, num_input_channels, num_segmentation_heads and
        enable_deep_supervision.
        """

        assert (
            self.nnunet_plans_bytes is not None
            and self.num_input_channels is not None
            and self.num_segmentation_heads is not None
            and self.enable_deep_supervision is not None
            and self.nnunet_config is not None
        )

        other_state_to_save = {
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

        self.checkpoint_and_state_module.save_state(
            state_checkpoint_name=self.state_checkpoint_name,
            server_parameters=self.parameters,
            other_state=other_state_to_save,
        )

    def _load_server_state(self) -> bool:
        """
        Load server checkpoint consisting of model, history, server name, current round and metrics reporter.
        The method overrides parent to add any necessary state when loading the checkpoint.
        """
        # Attempt to load the server state if it exists. This variable will be None if it does not.
        server_state = self.checkpoint_and_state_module.maybe_load_state(self.state_checkpoint_name)

        if server_state is None:
            return False

        # Standard attributes to load
        narrow_dict_type_and_set_attribute(self, server_state, "current_round", "current_round", int)
        narrow_dict_type_and_set_attribute(self, server_state, "server_name", "server_name", str)
        narrow_dict_type_and_set_attribute(self, server_state, "reports_manager", "reports_manager", ReportsManager)
        narrow_dict_type_and_set_attribute(self, server_state, "history", "history", History)
        narrow_dict_type_and_set_attribute(
            self, server_state, "model", "parameters", nn.Module, func=get_all_model_parameters
        )
        # Needed for when _hydrate_model_for_checkpointing is called
        narrow_dict_type_and_set_attribute(self, server_state, "model", "server_model", nn.Module)

        # NnunetServer specific attributes to load
        narrow_dict_type_and_set_attribute(self, server_state, "nnunet_plans_bytes", "nnunet_plans_bytes", bytes)
        narrow_dict_type_and_set_attribute(self, server_state, "num_segmentation_heads", "num_segmentation_heads", int)
        narrow_dict_type_and_set_attribute(self, server_state, "num_input_channels", "num_input_channels", int)
        narrow_dict_type_and_set_attribute(
            self, server_state, "enable_deep_supervision", "enable_deep_supervision", bool
        )
        narrow_dict_type_and_set_attribute(self, server_state, "nnunet_config", "nnunet_config", NnunetConfig)
        return True
