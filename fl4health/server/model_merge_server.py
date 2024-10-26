import datetime
import timeit
from logging import INFO, WARNING
from typing import Dict, Optional, Sequence, Tuple

import torch.nn as nn
from flwr.common.logger import log
from flwr.common.parameter import parameters_to_ndarrays
from flwr.common.typing import Scalar
from flwr.server.client_manager import ClientManager
from flwr.server.history import History
from flwr.server.server import Server
from flwr.server.strategy import Strategy

from fl4health.checkpointing.checkpointer import LatestTorchCheckpointer
from fl4health.parameter_exchange.parameter_exchanger_base import ParameterExchanger
from fl4health.reporting.base_reporter import BaseReporter
from fl4health.reporting.reports_manager import ReportsManager
from fl4health.strategies.model_merge_strategy import ModelMergeStrategy
from fl4health.utils.random import generate_hash


class ModelMergeServer(Server):
    def __init__(
        self,
        *,
        client_manager: ClientManager,
        strategy: Optional[Strategy] = None,
        checkpointer: Optional[LatestTorchCheckpointer] = None,
        server_model: Optional[nn.Module] = None,
        parameter_exchanger: Optional[ParameterExchanger] = None,
        reporters: Sequence[BaseReporter] | None = None,
        server_name: Optional[str] = None,
    ) -> None:
        """
        ModelMergeServer provides functionality to fetch client weights, perform a simple average,
            redistribute to clients for evaluation. Optionally can perform server side evaluation as well.
        Args:
            client_manager (ClientManager): Determines the mechanism by which clients are sampled by the server, if
                they are to be sampled at all.
            strategy (Optional[Strategy], optional): The aggregation strategy to be used by the server to handle
                client updates sent by the participating clients. Must be ModelMergeStrategy.
            checkpointer (Optional[LatestTorchCheckpointer], optional): To be provided if the server should perform
                server side checkpointing on the merged model. If none, then no server-side checkpointing is
                performed. Defaults to None.
            server_model (Optional[nn.Module]): Optional model to be hydrated with parameters from model merge if doing
                server side checkpointing. Must only be provided if checkpointer is also provided. Defaults to None.
            parameter_exchanger (Optional[ExchangerType]): Optional parameter exchanger to be used to hydrate the
                model. Only used if checkpointer and model are also not None. Defaults to None.
            reporters (Sequence[BaseReporter], optional): A sequence of FL4Health reporters which the server should
                send data to before and after each round.
            server_name (Optional[str]): An optional string name to uniquely identify server.
        """
        assert isinstance(strategy, ModelMergeStrategy)
        assert (server_model is None and checkpointer is None and parameter_exchanger is None) or (
            server_model is not None and checkpointer is not None and parameter_exchanger is not None
        )
        super().__init__(client_manager=client_manager, strategy=strategy)

        self.checkpointer = checkpointer
        self.server_model = server_model
        self.parameter_exchanger = parameter_exchanger
        self.server_name = server_name if server_name is not None else generate_hash()

        # Initialize reporters with server name information.
        self.reports_manager = ReportsManager(reporters)
        self.reports_manager.initialize(id=self.server_name)

    def fit(self, num_rounds: int, timeout: Optional[float]) -> Tuple[History, float]:
        """
        Performs a fit round in which the local client weights are evaluated on their test set,
            uploaded to the server and averaged, then redistributed to clients for evaluation.
            Optionally, can perform evaluation of the merged model on the server side as well.

        Args:
            num_rounds (int): Not used.
            timeout (Optional[float]): Timeout in seconds that the server should wait for the clients to respond.
                If none, then it will wait for the minimum number to respond indefinitely.

        Returns:
            Tuple[History, float]: The first element of the tuple is a History object containing the aggregated
                metrics returned from the clients. Tuple also contains elapsed time in seconds for round.
        """
        self.reports_manager.report({"host_type": "server", "fit_start": datetime.datetime.now()})

        history = History()

        # Run Federated Model Merging
        log(INFO, "Federated Model Merging Starting")
        start_time = timeit.default_timer()

        res_fit = self.fit_round(
            server_round=1,
            timeout=timeout,
        )

        if res_fit is not None:
            parameters_prime, fit_metrics, _ = res_fit  # fit_metrics_aggregated
            if parameters_prime:
                self.parameters = parameters_prime
            history.add_metrics_distributed_fit(server_round=1, metrics=fit_metrics)
        else:
            log(WARNING, "Federated Model Merging Failed")

        res_fed = self.evaluate_round(server_round=1, timeout=timeout)
        if res_fed is not None:
            # ignore loss as one is not defined in model merging
            _, evaluate_metrics_fed, _ = res_fed
            if evaluate_metrics_fed is not None:
                history.add_metrics_distributed(server_round=1, metrics=evaluate_metrics_fed)

        # Evaluate model using strategy implementation
        res_cen = self.strategy.evaluate(1, parameters=self.parameters)
        if res_cen is not None:
            # ignore loss as one is not defined in model merging
            _, metrics_cen = res_cen
            history.add_metrics_centralized(server_round=1, metrics=metrics_cen)

        # Checkpoint based on dummy loss aggregated and metrics aggregated since
        # we are using LatestTorchCheckpointer and will always checkpoint if
        # server_model, parameter_exchanger and checkpointer are not None
        self._maybe_checkpoint(loss_aggregated=0.0, metrics_aggregated={}, server_round=1)

        self.reports_manager.report(
            data={
                "fit_end": datetime.datetime.now(),
                "metrics_centralized": history.metrics_centralized,
                "losses_centralized": history.losses_centralized,
                "host_type": "server",
            }
        )

        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        log(INFO, "Federated Model Merging Finished in %s", elapsed)
        return history, elapsed

    def _hydrate_model_for_checkpointing(self) -> nn.Module:
        """
        Method used for converting server parameters into a torch model that can be checkpointed.

        Returns:
            nn.Module: Torch model to be checkpointed by a torch checkpointer.
        """
        assert self.server_model is not None and self.parameter_exchanger is not None
        model_ndarrays = parameters_to_ndarrays(self.parameters)
        self.parameter_exchanger.pull_parameters(model_ndarrays, self.server_model)
        return self.server_model

    def _maybe_checkpoint(
        self, loss_aggregated: float, metrics_aggregated: Dict[str, Scalar], server_round: int
    ) -> None:
        """
        Method to checkpoint merged model on server side if the checkpointer, server_model and
            parameter_exchanger provided at initialization are all not None.

        Args:
            loss_aggregated (float): Not used.
            metrics_aggregated (Dict[str, Scalar]): Not used.
            server_round (int): Not used.
        """
        if self.checkpointer and self.server_model and self.parameter_exchanger:
            model = self._hydrate_model_for_checkpointing()
            self.checkpointer.maybe_checkpoint(model, loss_aggregated, metrics_aggregated)
        else:
            attribute_dict = {
                "checkpointer": self.checkpointer,
                "server_model": self.server_model,
                "parameter_exchanger": self.parameter_exchanger,
            }

            error_str = " and ".join([key for key, val in attribute_dict.items() if val is None])
            log(
                WARNING,
                f"""All of checkpointer, server_model and parameter_exchanger must be None to
                perform server-side checkpointing. {error_str} is None""",
            )
