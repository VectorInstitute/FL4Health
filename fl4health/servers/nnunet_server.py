import pickle
import warnings
from collections.abc import Callable, Sequence
from logging import INFO
from typing import Any

from flwr.common import Parameters
from flwr.common.logger import log
from flwr.common.typing import Code, Config, EvaluateIns, FitIns, GetPropertiesIns, Scalar
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy

from fl4health.checkpointing.server_module import NnUnetServerCheckpointAndStateModule
from fl4health.reporting.base_reporter import BaseReporter
from fl4health.servers.base_server import FlServer
from fl4health.utils.config import narrow_dict_type
from fl4health.utils.nnunet_utils import NnunetConfig


with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

FIT_CFG_FN = Callable[[int, Parameters, ClientManager], list[tuple[ClientProxy, FitIns]]]
EVAL_CFG_FN = Callable[[int, Parameters, ClientManager], list[tuple[ClientProxy, EvaluateIns]]]
CFG_FN = FIT_CFG_FN | EVAL_CFG_FN


def add_items_to_config_fn(fn: CFG_FN, items: Config) -> CFG_FN:
    """
    Accepts a flwr Strategy configure function (either ``configure_fit`` or ``configure_evaluate``) and returns a new
    function  that returns the same thing except the dictionary items in the items argument have been added to the
    config that  is returned by the original function.

    Args:
        fn (CFG_FN): The Strategy configure function to wrap
        items (Config): A ``Config`` containing additional items to update the original config with

    Returns:
        (CFG_FN): The wrapped function. Argument and return type is the same
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
        on_init_parameters_config_fn: Callable[[int], dict[str, Scalar]],
        strategy: Strategy | None = None,
        reporters: Sequence[BaseReporter] | None = None,
        checkpoint_and_state_module: NnUnetServerCheckpointAndStateModule | None = None,
        server_name: str | None = None,
        accept_failures: bool = True,
        nnunet_trainer_class: type[nnUNetTrainer] = nnUNetTrainer,
        global_deep_supervision: bool = False,
    ) -> None:
        """
        A Basic ``FlServer`` with added functionality to ask a client to initialize the global nnunet plans if one was
        not provided in the config. Intended for use with ``NnUNetClient``.

        Args:
            client_manager (ClientManager): Determines the mechanism by which clients are sampled by the server, if
                they are to be sampled at all.
            fl_config (Config): This should be the configuration that was used to setup the federated training.
                In most cases it should be the "source of truth" for how FL training/evaluation should proceed. For
                example, the config used to produce the ``on_fit_config_fn`` and ``on_evaluate_config_fn`` for the
                strategy.

                **NOTE**: This config is **DISTINCT** from the Flwr server config, which is extremely minimal.
            on_init_parameters_config_fn (Callable[[int], dict[str, Scalar]]): Function used to configure how one
                asks a client to provide parameters from which to initialize all other clients by providing a
                ``Config`` dictionary. For ``NnunetServers`` this is a required function to provide the additional
                information necessary to a client for parameter initialization
            strategy (Strategy | None, optional): The aggregation strategy to be used by the server to handle
                client updates and other information potentially sent by the participating clients. If None the
                strategy is FedAvg as set by the flwr Server. Defaults to None.
            reporters (Sequence[BaseReporter] | None, optional): A sequence of FL4Health reporters which the client
                should send data to. Defaults to None.
            checkpoint_and_state_module (NnUnetServerCheckpointAndStateModule | None, optional): This module is used
                to handle both model checkpointing and state checkpointing. The former is aimed at saving model
                artifacts to be used or evaluated after training. The latter is used to preserve training state
                (including models) such that if FL training is interrupted, the process may be restarted. If no
                module is provided, no checkpointing or state preservation will happen. Defaults to None.

                **NOTE**: For NnUnet, this module is allowed to have all components defined other than the model, as it
                may be set later when the server asks the clients to provide the architecture.
            server_name (str | None, optional): An optional string name to uniquely identify server. This name is also
                used as part of any state checkpointing done by the server. Defaults to None.
            accept_failures (bool, optional): Determines whether the server should accept failures during training or
                evaluation from clients or not. If set to False, this will cause the server to shutdown all clients
                and throw an exception. Defaults to True.
            nnunet_trainer_class (type[nnUNetTrainer]): ``nnUNetTrainer`` class.
                Useful for passing custom ``nnUNetTrainer``. Defaults to the standard ``nnUNetTrainer`` class.
                Must match the ``nnunet_trainer_class`` passed to the ``NnunetClient``.
            global_deep_supervision (bool): Whether or not the global model should use deep supervision. Does
                not affect the model architecture just the output during inference. This argument applies only to the
                global model, not local client models. Defaults to False.
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
        self.global_deep_supervision = global_deep_supervision
        self.nnunet_config = NnunetConfig(self.fl_config["nnunet_config"])

        self.nnunet_plans_bytes: bytes
        self.num_input_channels: int
        self.num_segmentation_heads: int

    def initialize_server_model(self) -> None:
        """Initializes the global server model so that it can be checkpointed."""
        # Ensure required attributes are set
        assert (
            self.nnunet_plans_bytes is not None
            and self.num_input_channels is not None
            and self.num_segmentation_heads is not None
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
            self.global_deep_supervision,
        )
        self.checkpoint_and_state_module.model = model

    def update_before_fit(self, num_rounds: int, timeout: float | None) -> None:
        """
        Hook method to allow the server to do some additional initialization prior to fitting.

        ``NnunetServer`` uses this method to sample a client for properties for one of two reasons

        1. If a global ``nnunet_plans`` file is not provided in the config, this method will request that a random
           client which generate a plans file from it local dataset and return it to the server through the
           ``get_properties`` RPC. The server then distributes the ``nnunet_plans`` to the other clients by including
           it in the config for subsequent FL rounds.

           AND/OR

        2. If server side state or model checkpointing is being used, then server will  poll a client in order to have
           the required properties to instantiate the model architecture on the server side. These properties include
           ``num_segmentation_heads`` and ``num_input_channels``, essentially the number of input and output channels
           (which are not specified in nnunet plans for some reason).

        Args:
            num_rounds (int): The number of server rounds of FL to be performed.
            timeout (float | None, optional): The server's timeout parameter. Useful if one is requesting
                information from a client. Defaults to None, which indicates indefinite timeout.
        """
        # Check if nnunet_plans specified config returned by configure_fit
        dummy_params = Parameters([], "None")
        config = self.strategy.configure_fit(0, dummy_params, self._client_manager)[0][1].config
        plans_bytes = config.get("nnunet_plans")

        # Check for checkpointers
        checkpointer_exists = (
            self.checkpoint_and_state_module.state_checkpointer is not None
            or self.checkpoint_and_state_module.model_checkpointers is not None
        )

        if checkpointer_exists or plans_bytes is None:
            log(INFO, "")
            log(INFO, "[PRE-INIT]")
            log(INFO, "Requesting properties from one random client via get_properties")

            # 1) If nnUnet plans are unspecified, we ask a client to generate the global plans using its local dataset
            if plans_bytes is None:
                log(INFO, "\tThis client will be asked to initialize the global nnunet plans")

            # 2) If the checkpointer is not None, then we want to do checkpointing. Therefore we need to
            #   be able to construct the model and for that we need the number of input and output channels.
            if checkpointer_exists:
                log(
                    INFO,
                    "\tThis client's local dataset will be used to determine the number of input and output channels",
                )

            # Sample a random client and request properties
            random_client = self._client_manager.sample(1)[0]
            ins = GetPropertiesIns(config=config)
            properties_res = random_client.get_properties(ins=ins, timeout=timeout, group_id=0)

            if properties_res.status.code == Code.OK:
                log(INFO, "Received properties from one random client")
            else:
                raise Exception("Failed to successfully receive properties from client")
            properties = properties_res.properties

            # Set self.nnunet_plans_bytes
            if plans_bytes is None:
                self.nnunet_plans_bytes = narrow_dict_type(properties, "nnunet_plans", bytes)
            else:
                assert isinstance(plans_bytes, bytes)
                self.nnunet_plans_bytes = plans_bytes

            # Save number of input and output channels as attributes
            self.num_segmentation_heads = narrow_dict_type(properties, "num_segmentation_heads", int)
            self.num_input_channels = narrow_dict_type(properties, "num_input_channels", int)

            # Initialize global model
            if checkpointer_exists:
                self.initialize_server_model()

            # If the state_checkpointer has been specified and a state checkpoint exists, the state
            # will be loaded when executing ``fit_with_per_round_checkpointing`` of the base_server.
            # NOTE: Inherent assumption that if checkpoint exists for server that it also will exist for client.

            # Wrap config functions so that we are sure the nnunet_plans are included
            new_fit_cfg_fn = add_items_to_config_fn(
                self.strategy.configure_fit, {"nnunet_plans": self.nnunet_plans_bytes}
            )
            new_eval_cfg_fn = add_items_to_config_fn(
                self.strategy.configure_evaluate, {"nnunet_plans": self.nnunet_plans_bytes}
            )
            self.strategy.configure_fit = new_fit_cfg_fn  # type: ignore
            self.strategy.configure_evaluate = new_eval_cfg_fn  # type: ignore

        # Finish
        log(INFO, "")

    def _save_server_state(self) -> None:
        """
        Save server checkpoint consisting of model, history, server round, metrics reporter and server name. This
        method overrides parent to also `checkpoint` ``nnunet_plans``, ``num_input_channels``,
        ``num_segmentation_heads`` and ``global_deep_supervision``.
        """
        assert (
            self.nnunet_plans_bytes is not None
            and self.num_input_channels is not None
            and self.num_segmentation_heads is not None
            and self.global_deep_supervision is not None
            and self.nnunet_config is not None
        )

        super()._save_server_state()
