from logging import INFO, WARN
from typing import Any, Callable, List, Optional, Tuple, Union

from flwr.common import Parameters
from flwr.common.logger import log
from flwr.common.typing import Code, Config, EvaluateIns, FitIns, GetPropertiesIns
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from fl4health.server.base_server import FlServerWithInitializer

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


class NnUNetServer(FlServerWithInitializer):
    """
    A Basic FlServer with added functionality to ask a client to initialize
    the global nnunet plans if one was not provided in the config. Intended
    for use with NnUNetClient
    """

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
            log(WARN, "Failed to receive properties from client to initialize nnnunet plans")

        properties = properties_res.properties

        # NnUNetClient has serialized nnunet_plans as a property
        plans_bytes = properties["nnunet_plans"]

        # Wrap config functions so that nnunet_plans is included
        new_fit_cfg_fn = add_items_to_config_fn(self.strategy.configure_fit, {"nnunet_plans": plans_bytes})
        new_eval_cfg_fn = add_items_to_config_fn(self.strategy.configure_evaluate, {"nnunet_plans": plans_bytes})
        setattr(self.strategy, "configure_fit", new_fit_cfg_fn)
        setattr(self.strategy, "configure_evaluate", new_eval_cfg_fn)

        # Finish
        self.initialized = True
        log(INFO, "")
