import random
import warnings
from functools import partial
from logging import INFO
from typing import Callable, List, Optional, Tuple

from flwr.common.logger import log
from flwr.common.typing import Config
from flwr.server.client_manager import ClientManager
from flwr.server.history import History

from fl4health.checkpointing.checkpointer import TorchCheckpointer
from fl4health.feature_alignment.constants import (
    CURRENT_SERVER_ROUND,
    FEATURE_INFO,
    FORMAT_SPECIFIED,
    INPUT_DIMENSION,
    OUTPUT_DIMENSION,
)
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
        strategy: BasicFedAvg,
        wandb_reporter: Optional[ServerWandBReporter] = None,
        checkpointer: Optional[TorchCheckpointer] = None,
        tab_features_info: Optional[TabFeaturesInfoEncoder] = None,
    ) -> None:
        assert isinstance(strategy, BasicFedAvg)
        if strategy.on_fit_config_fn is not None:
            warnings.warn("strategy.on_fit_config_fn will be overwritten.")
        if strategy.initial_parameters is not None:
            warnings.warn("strategy.initial_parameters will be overwritten.")

        super().__init__(client_manager, strategy, wandb_reporter, checkpointer)
        # The server performs one or two rounds of polls before the normal federated training.
        # The first one gathers feature information if the server does not already have it,
        # and the second one gathers the input/output dimensions of the model.
        self.initial_polls_complete = False
        self.tab_features_info = tab_features_info
        self.config = config
        self.initialize_parameters = initialize_parameters
        self.format_info_gathered = False
        # casting self.strategy to BasicFedAvg so its on_fit_config_fn can be specified.
        self.strategy: BasicFedAvg
        self.strategy.on_fit_config_fn = partial(fit_config, self.config, self.format_info_gathered)

    def fit(self, num_rounds: int, timeout: Optional[float]) -> History:
        """Run federated averaging for a number of rounds."""

        assert isinstance(self.strategy, BasicFedAvg)

        # Before the normal fitting round begins, the server provides all clients
        # the feature information needed to perform feature alignment. Then the server
        # gathers information from the clients that is necessary for initializing the global model.
        if not self.initial_polls_complete:

            # If the server does not have the needed feature info a priori,
            # then it requests such information from the clients before the
            # very first fitting round.
            if self.tab_features_info is None:
                # A random client's feature information is selected as the source of truth for feature alignment.
                feature_info = self.poll_clients_for_feature_info(timeout)

                rand_idx = random.randint(0, len(feature_info) - 1)

                feature_info_source = feature_info[rand_idx]
            # If the server already has the feature info, then it simply sends it to the clients.
            else:
                log(
                    INFO,
                    "Features information source already specified. Sending to clients to perform feature alignment.",
                )
                feature_info_source = self.tab_features_info.to_json()

            # the feature information is sent to clients through the config parameter.
            self.config[FEATURE_INFO] = feature_info_source
            self.format_info_gathered = True

            self.strategy.on_fit_config_fn = partial(fit_config, self.config, self.format_info_gathered)

            # Now the server waits until feature alignment is performed on the clients' side
            # and subsequently requests the input and output dimensions, which are needed for initializing
            # the global model.
            input_dimension, output_dimension = self.poll_clients_for_dimension_info(timeout)
            log(INFO, f"input dimension: {input_dimension}, output dimension: {output_dimension}")
            self.strategy.initial_parameters = self.initialize_parameters(input_dimension, output_dimension)
            self.initial_polls_complete = True

        # Normal federated learning rounds commence after all clients' features
        # are aligned and global model is initialized.
        return super().fit(num_rounds=num_rounds, timeout=timeout)

    def poll_clients_for_feature_info(self, timeout: Optional[float]) -> List[str]:
        log(INFO, "Feature information source unspecified. Polling clients for feature information.")
        assert isinstance(self.strategy, BasicFedAvg)
        client_instructions = self.strategy.configure_poll(server_round=1, client_manager=self._client_manager)
        results, _ = poll_clients(
            client_instructions=client_instructions, max_workers=self.max_workers, timeout=timeout
        )

        feature_info: List[str] = [
            str(get_properties_res.properties[FEATURE_INFO]) for (_, get_properties_res) in results
        ]

        return feature_info

    def poll_clients_for_dimension_info(self, timeout: Optional[float]) -> Tuple[int, int]:
        log(INFO, "Waiting for Clients to align features and then polling for dimension information.")
        assert isinstance(self.strategy, BasicFedAvg)
        client_instructions = self.strategy.configure_poll(server_round=1, client_manager=self._client_manager)
        results, _ = poll_clients(
            client_instructions=client_instructions, max_workers=self.max_workers, timeout=timeout
        )

        input_dimension = int(results[0][1].properties[INPUT_DIMENSION])
        out_put_dimension = int(results[0][1].properties[OUTPUT_DIMENSION])

        return input_dimension, out_put_dimension


def fit_config(config: Config, format_specified: bool, current_server_round: int) -> Config:
    config[FORMAT_SPECIFIED] = format_specified
    config[CURRENT_SERVER_ROUND] = current_server_round
    return config
