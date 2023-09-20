import random
import warnings
from functools import partial
from logging import INFO
from typing import Callable, List, Optional, Tuple

from flwr.common.logger import log
from flwr.common.typing import Config
from flwr.server.client_manager import ClientManager
from flwr.server.history import History
from flwr.server.strategy import Strategy

from fl4health.checkpointing.checkpointer import TorchCheckpointer
from fl4health.feature_alignment.tab_features_info_encoder import TabFeaturesInfoEncoder
from fl4health.reporting.fl_wanb import ServerWandBReporter
from fl4health.server.base_server import FlServer
from fl4health.server.polling import poll_clients
from fl4health.strategies.basic_fedavg import BasicFedAvg


class TabularFeatureAlignmentServer(FlServer):
    """
    This server is used when the clients all have tabular data that needs to be
    aligned.

    tab_features_info: the information that is required for aligning client features.
    If it is not specified, then the server will randomly poll a client and gather
    this information from its data source.
    """

    def __init__(
        self,
        client_manager: ClientManager,
        config: Config,
        initialize_parameters: Callable,
        strategy: Optional[Strategy] = None,
        wandb_reporter: Optional[ServerWandBReporter] = None,
        checkpointer: Optional[TorchCheckpointer] = None,
        tab_features_info: Optional[TabFeaturesInfoEncoder] = None,
    ) -> None:
        assert isinstance(self.strategy, BasicFedAvg)
        if self.strategy.on_fit_config_fn is not None:
            warnings.warn("self.strategy.on_fit_config_fn will be overwritten.")
        if self.strategy.initial_parameters is not None:
            warnings.warn("self.strategy.initial_parameters will be overwritten.")

        super().__init__(client_manager, strategy, wandb_reporter, checkpointer)
        self.initial_polls_complete = False
        self.tab_features_info = tab_features_info
        self.config = config
        self.initialize_parameters = initialize_parameters

    def fit(self, num_rounds: int, timeout: Optional[float]) -> History:
        """Run federated averaging for a number of rounds."""

        assert isinstance(self.strategy, BasicFedAvg)

        # Before the normal fitting round commences, the server provides all clients
        # the feature information needed to perform feature alignment. Then the server
        # gathers information from the clients necessary to initialize the global model.
        if not self.initial_polls_complete:

            # If the server does not have the needed feature info,
            # then it requests such information from the clients before the
            # very first fitting round.
            if self.tab_features_info is None:
                # A random client's feature information is selected as the standard for feature alignment.
                feature_info = self.poll_clients_for_feature_info(timeout)

                rand_idx = random.randint(0, len(feature_info) - 1)

                feature_info_source = feature_info[rand_idx]
            else:
                feature_info_source = self.tab_features_info.to_json()

            # the feature information is sent to clients through the config parameter.
            self.config["feature_info"] = feature_info_source

            def fit_config(config: Config, current_round: int) -> Config:
                config["format_specified"] = current_round > 1
                return config

            self.strategy.on_fit_config_fn = partial(fit_config, self.config)

            input_dimension, output_dimension = self.poll_clients_for_dimension_info(timeout)
            self.strategy.initial_parameters = self.initialize_parameters(input_dimension, output_dimension)
            self.initial_polls_complete = True

        return super().fit(num_rounds=num_rounds, timeout=timeout)

    def poll_clients_for_feature_info(self, timeout: Optional[float]) -> List[str]:
        log(INFO, "Polling Clients for feature information")
        assert isinstance(self.strategy, BasicFedAvg)
        client_instructions = self.strategy.configure_poll(server_round=1, client_manager=self._client_manager)
        results, _ = poll_clients(
            client_instructions=client_instructions, max_workers=self.max_workers, timeout=timeout
        )

        feature_info: List[str] = [
            str(get_properties_res.properties["feature_info"]) for (_, get_properties_res) in results
        ]

        return feature_info

    def poll_clients_for_dimension_info(self, timeout: Optional[float]) -> Tuple[int, int]:
        log(INFO, "Polling Clients for dimension information")
        assert isinstance(self.strategy, BasicFedAvg)
        client_instructions = self.strategy.configure_poll(server_round=1, client_manager=self._client_manager)
        results, _ = poll_clients(
            client_instructions=client_instructions, max_workers=self.max_workers, timeout=timeout
        )

        input_dimension = int(results[0][1].properties["input_dimension"])
        target_dimension = int(results[0][1].properties["target_dimension"])

        return input_dimension, target_dimension
