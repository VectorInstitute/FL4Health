import random
from collections.abc import Callable, Sequence
from functools import partial
from logging import DEBUG, INFO, WARNING

from flwr.common import Parameters
from flwr.common.logger import log
from flwr.common.typing import Config, Scalar
from flwr.server.client_manager import ClientManager
from flwr.server.history import History

from fl4health.checkpointing.server_module import BaseServerCheckpointAndStateModule
from fl4health.feature_alignment.constants import (
    CURRENT_SERVER_ROUND,
    FEATURE_INFO,
    INPUT_DIMENSION,
    OUTPUT_DIMENSION,
    SOURCE_SPECIFIED,
)
from fl4health.feature_alignment.tab_features_info_encoder import TabularFeaturesInfoEncoder
from fl4health.reporting.base_reporter import BaseReporter
from fl4health.servers.base_server import FlServer
from fl4health.servers.polling import poll_clients
from fl4health.strategies.basic_fedavg import BasicFedAvg


class TabularFeatureAlignmentServer(FlServer):
    def __init__(
        self,
        client_manager: ClientManager,
        config: Config,
        initialize_parameters: Callable[[int, int], Parameters],
        strategy: BasicFedAvg,
        tabular_features_source_of_truth: TabularFeaturesInfoEncoder | None = None,
        reporters: Sequence[BaseReporter] | None = None,
        checkpoint_and_state_module: BaseServerCheckpointAndStateModule | None = None,
        on_init_parameters_config_fn: Callable[[int], dict[str, Scalar]] | None = None,
        server_name: str | None = None,
        accept_failures: bool = True,
    ) -> None:
        """
        This server is used when the clients all have tabular data that needs to be aligned.

        Args:
            client_manager (ClientManager): Determines the mechanism by which clients are sampled by the server, if
                they are to be sampled at all.
            config (Config): This should be the configuration that was used to setup the federated alignment.
                In most cases it should be the "source of truth" for how FL alignment should proceed.

                **NOTE**: This config is **DISTINCT** from the Flwr server config, which is extremely minimal.
            initialize_parameters (Callable[[int, int], Parameters]): Function used to initialize the model to be
                trained and used for the tabular task.

                **NOTE**: The model architecture is not finalized until we are able to determine the dimensionality of
                the input and output space during feature alignment.
            strategy (BasicFedAvg): The aggregation strategy to be used by the server to handle.
                client updates and other information potentially sent by the participating clients. If None the
                strategy is FedAvg as set by the flwr Server.
            tabular_features_source_of_truth (TabularFeaturesInfoEncoder | None, optional): The information that is
                required for aligning client features. If it is not specified, then the server will randomly poll a
                client and gather this information from its data source. Defaults to None.
            reporters (Sequence[BaseReporter] | None, optional): Sequence of FL4Health reporters which the server
                should send data to before and after each round. Defaults to None
            checkpoint_and_state_module (BaseServerCheckpointAndStateModule | None, optional): This module is used
                to handle both model checkpointing and state checkpointing. The former is aimed at saving model
                artifacts to be used or evaluated after training. The latter is used to preserve training state
                (including models) such that if FL training is interrupted, the process may be restarted. If no
                module is provided, no checkpointing or state preservation will happen. Defaults to None.
            on_init_parameters_config_fn (Callable[[int], dict[str, Scalar]] | None, optional): Function used to
                configure how one asks a client to provide parameters from which to initialize all other clients by
                providing a ``Config`` dictionary. If this is none, then a blank config is sent with the parameter
                request (which is default behavior for flower servers). Defaults to None.
            server_name (str | None, optional): An optional string name to uniquely identify server. This name is also
                used as part of any state checkpointing done by the server. Defaults to None.
            accept_failures (bool, optional): Determines whether the server should accept failures during training or
                evaluation from clients or not. If set to False, this will cause the server to shutdown all clients
                and throw an exception. Defaults to True.
        """
        if strategy.on_fit_config_fn is not None:
            log(WARNING, "strategy.on_fit_config_fn will be overwritten.")
        if strategy.initial_parameters is not None:
            log(WARNING, "strategy.initial_parameters will be overwritten.")

        super().__init__(
            client_manager=client_manager,
            fl_config=config,
            strategy=strategy,
            reporters=reporters,
            checkpoint_and_state_module=checkpoint_and_state_module,
            on_init_parameters_config_fn=on_init_parameters_config_fn,
            server_name=server_name,
            accept_failures=accept_failures,
        )
        # The server performs one or two rounds of polls before the normal federated training.
        # The first one gathers feature information if the server does not already have it,
        # and the second one gathers the input/output dimensions of the model.
        self.initial_polls_complete = False
        self.tab_features_info = tabular_features_source_of_truth
        self.initialize_parameters = initialize_parameters
        self.source_info_gathered = False
        self.dimension_info: dict[str, int] = {}
        # ensure that self.strategy has type BasicFedAvg so its on_fit_config_fn can be specified.
        assert isinstance(self.strategy, BasicFedAvg), "This server is only compatible with BasicFedAvg at this time"
        self.strategy.on_fit_config_fn = partial(fit_config, self.fl_config, self.source_info_gathered)

    def _set_dimension_info(self, input_dimension: int, output_dimension: int) -> None:
        self.dimension_info[INPUT_DIMENSION] = input_dimension
        self.dimension_info[OUTPUT_DIMENSION] = output_dimension

    def _get_initial_parameters(self, server_round: int, timeout: float | None) -> Parameters:
        assert INPUT_DIMENSION in self.dimension_info and OUTPUT_DIMENSION in self.dimension_info
        input_dimension = self.dimension_info[INPUT_DIMENSION]
        output_dimension = self.dimension_info[OUTPUT_DIMENSION]
        return self.initialize_parameters(input_dimension, output_dimension)

    def fit(self, num_rounds: int, timeout: float | None) -> tuple[History, float]:
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
            self.fl_config[FEATURE_INFO] = feature_info_source
            self.source_info_gathered = True

            self.strategy.on_fit_config_fn = partial(fit_config, self.fl_config, self.source_info_gathered)

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

    def poll_clients_for_feature_info(self, timeout: float | None) -> str:
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
        return str(get_properties_res.properties[FEATURE_INFO])

    def poll_clients_for_dimension_info(self, timeout: float | None) -> tuple[int, int]:
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


def fit_config(config: Config, source_specified: bool, current_server_round: int) -> Config:
    config[SOURCE_SPECIFIED] = source_specified
    config[CURRENT_SERVER_ROUND] = current_server_round
    return config
