import random
from functools import partial
from logging import DEBUG, INFO, WARNING
from typing import Callable, Dict, Optional, Tuple

from flwr.common import Parameters
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
from fl4health.feature_alignment.tab_features_info_encoder import TabularFeaturesInfoEncoder
from fl4health.reporting.fl_wanb import ServerWandBReporter
from fl4health.server.base_server import FlServer
from fl4health.server.polling import poll_clients
from fl4health.strategies.basic_fedavg import BasicFedAvg


class TabularFeatureAlignmentServer(FlServer):
    """
    This server is used when the clients all have tabular data that needs to be
    aligned.

    Args:
        client_manager (ClientManager): Determines the mechanism by which clients are sampled by the server, if
                they are to be sampled at all.
        strategy (Optional[Strategy], optional): The aggregation strategy to be used by the server to handle.
            client updates and other information potentially sent by the participating clients. If None the
            strategy is FedAvg as set by the flwr Server.
        wandb_reporter (Optional[ServerWandBReporter], optional): To be provided if the server is to log
            information and results to a Weights and Biases account. If None is provided, no logging occurs.
            Defaults to None.
        checkpointer (Optional[TorchCheckpointer], optional): To be provided if the server should perform
            server side checkpointing based on some criteria. If none, then no server-side checkpointing is
            performed. Defaults to None.
        tab_features_source_of_truth (Optional[TabularFeaturesInfoEncoder]): The information that is required
        for aligning client features. If it is not specified, then the server will randomly poll a client
        and gather this information from its data source.
    """

    def __init__(
        self,
        client_manager: ClientManager,
        config: Config,
        initialize_parameters: Callable[..., Parameters],
        strategy: BasicFedAvg,
        wandb_reporter: Optional[ServerWandBReporter] = None,
        checkpointer: Optional[TorchCheckpointer] = None,
        tabular_features_source_of_truth: Optional[TabularFeaturesInfoEncoder] = None,
    ) -> None:
        if strategy.on_fit_config_fn is not None:
            log(WARNING, "strategy.on_fit_config_fn will be overwritten.")
        if strategy.initial_parameters is not None:
            log(WARNING, "strategy.initial_parameters will be overwritten.")

        super().__init__(client_manager, strategy, wandb_reporter, checkpointer)
        # The server performs one or two rounds of polls before the normal federated training.
        # The first one gathers feature information if the server does not already have it,
        # and the second one gathers the input/output dimensions of the model.
        self.initial_polls_complete = False
        self.tab_features_info = tabular_features_source_of_truth
        self.config = config
        self.initialize_parameters = initialize_parameters
        self.format_info_gathered = False
        self.dimension_info: Dict[str, int] = {}
        # ensure that self.strategy has type BasicFedAvg so its on_fit_config_fn can be specified.
        assert isinstance(self.strategy, BasicFedAvg)
        self.strategy.on_fit_config_fn = partial(fit_config, self.config, self.format_info_gathered)

    def _set_dimension_info(self, input_dimension: int, output_dimension: int) -> None:
        self.dimension_info[INPUT_DIMENSION] = input_dimension
        self.dimension_info[OUTPUT_DIMENSION] = output_dimension

    def _get_initial_parameters(self, timeout: Optional[float]) -> Parameters:
        assert INPUT_DIMENSION in self.dimension_info and OUTPUT_DIMENSION in self.dimension_info
        input_dimension = self.dimension_info[INPUT_DIMENSION]
        output_dimension = self.dimension_info[OUTPUT_DIMENSION]
        return self.initialize_parameters(input_dimension, output_dimension)

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
                feature_info_source = self.poll_clients_for_feature_info(timeout)
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
            log(DEBUG, f"input dimension: {input_dimension}, output dimension: {output_dimension}")
            self._set_dimension_info(input_dimension, output_dimension)
            self.initial_polls_complete = True

        # Normal federated learning rounds commence after all clients' features
        # are aligned and global model is initialized.
        return super().fit(num_rounds=num_rounds, timeout=timeout)

    def poll_clients_for_feature_info(self, timeout: Optional[float]) -> str:
        log(INFO, "Feature information source unspecified. Polling clients for feature information.")
        assert isinstance(self.strategy, BasicFedAvg)
        client_instructions = self.strategy.configure_poll(server_round=1, client_manager=self._client_manager)
        # Randomly select one client to obtain its feature information.
        client_instructions_rand_sample = random.sample(population=client_instructions, k=1)
        results, _ = poll_clients(
            client_instructions=client_instructions_rand_sample, max_workers=self.max_workers, timeout=timeout
        )

        assert len(results) == 1
        _, get_properties_res = results[0]
        feature_info = str(get_properties_res.properties[FEATURE_INFO])
        return feature_info

    def poll_clients_for_dimension_info(self, timeout: Optional[float]) -> Tuple[int, int]:
        log(INFO, "Waiting for Clients to align features and then polling for dimension information.")
        assert isinstance(self.strategy, BasicFedAvg)
        client_instructions = self.strategy.configure_poll(server_round=1, client_manager=self._client_manager)

        # Since the features of all clients are aligned, we just select one client
        # to obtain the input/output dimensions.
        results, _ = poll_clients(
            client_instructions=client_instructions[:1], max_workers=self.max_workers, timeout=timeout
        )
        assert len(results) == 1
        input_dimension = int(results[0][1].properties[INPUT_DIMENSION])
        output_dimension = int(results[0][1].properties[OUTPUT_DIMENSION])

        return input_dimension, output_dimension


def fit_config(config: Config, format_specified: bool, current_server_round: int) -> Config:
    config[FORMAT_SPECIFIED] = format_specified
    config[CURRENT_SERVER_ROUND] = current_server_round
    return config
