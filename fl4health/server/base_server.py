from logging import INFO, WARNING
from typing import Dict, Generic, List, Optional, Tuple, TypeVar

import torch.nn as nn
from flwr.common.logger import log
from flwr.common.parameter import parameters_to_ndarrays
from flwr.common.typing import Scalar
from flwr.server.client_manager import ClientManager
from flwr.server.history import History
from flwr.server.server import EvaluateResultsAndFailures, Server
from flwr.server.strategy import Strategy

from fl4health.checkpointing.checkpointer import TorchCheckpointer
from fl4health.parameter_exchange.parameter_exchanger_base import ParameterExchanger
from fl4health.reporting.fl_wanb import ServerWandBReporter
from fl4health.server.polling import poll_clients
from fl4health.strategies.strategy_with_poll import StrategyWithPolling


class FlServer(Server):
    def __init__(
        self,
        client_manager: ClientManager,
        strategy: Optional[Strategy] = None,
        wandb_reporter: Optional[ServerWandBReporter] = None,
        checkpointer: Optional[TorchCheckpointer] = None,
    ) -> None:
        """
        Base Server for the library to facilitate strapping additional/useful machinery to the base flwr server.

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
        """
        super().__init__(client_manager=client_manager, strategy=strategy)
        self.wandb_reporter = wandb_reporter
        self.checkpointer = checkpointer

    def fit(self, num_rounds: int, timeout: Optional[float]) -> History:
        history = super().fit(num_rounds, timeout)
        if self.wandb_reporter:
            # report history to W and B
            self.wandb_reporter.report_metrics(num_rounds, history)
        return history

    def shutdown(self) -> None:
        if self.wandb_reporter:
            self.wandb_reporter.shutdown_reporter()

    def _hydrate_model_for_checkpointing(self) -> nn.Module:
        """
        This function is used for converting server parameters into a torch model that can be checkpointed. Note that
        if an inheriting class wants to do server-side checkpointing this functionality needs to be defined there.

        Raises:
            NotImplementedError: If this is called by a child class and the behavior is not defined, we throw an error.

        Returns:
            nn.Module: Should return a torch model to be checkpointed by a torch checkpointer.
        """
        # This function is used for converting server parameters into a torch model that can be checkpointed
        raise NotImplementedError()

    def _maybe_checkpoint(self, checkpoint_metric: float, server_round: int) -> None:
        if self.checkpointer:
            try:
                model = self._hydrate_model_for_checkpointing()
                self.checkpointer.maybe_checkpoint(model, checkpoint_metric)
            except NotImplementedError:
                # Checkpointer is defined but there is no server-side model hydration to produce a model from the
                # server state. This is not a deal breaker, but may be unintended behavior and the user will be warned
                if server_round == 1:
                    # just log message on the first round
                    log(
                        WARNING,
                        "Server model hydration is not defined but checkpointer is defined. Not checkpointing "
                        "model. Please ensure that this is intended",
                    )
        elif server_round == 1:
            # No checkpointer, just log message on the first round
            log(INFO, "No checkpointer present. Models will not be checkpointed on server-side.")

    def poll_clients_for_sample_counts(self, timeout: Optional[float]) -> List[int]:
        """
        Poll clients for sample counts from their training set, if you want to use this functionality your strategy
        needs to inherit from the StrategyWithPolling ABC and implement a configure_poll function.

        Args:
            timeout (Optional[float]): Timeout for how long the server will wait for clients to report counts. If none
                then the server waits indefinitely.

        Returns:
            List[int]: The number of training samples held by each client in the pool of available clients.
        """
        # Poll clients for sample counts, if you want to use this functionality your strategy needs to inherit from
        # the StrategyWithPolling ABC and implement a configure_poll function
        log(INFO, "Polling Clients for sample counts")
        assert isinstance(self.strategy, StrategyWithPolling)
        client_instructions = self.strategy.configure_poll(server_round=1, client_manager=self._client_manager)
        results, _ = poll_clients(
            client_instructions=client_instructions, max_workers=self.max_workers, timeout=timeout
        )

        sample_counts: List[int] = [
            int(get_properties_res.properties["num_train_samples"]) for (_, get_properties_res) in results
        ]
        log(INFO, f"Polling complete: Retrieved {len(sample_counts)} sample counts")

        return sample_counts

    def evaluate_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[Tuple[Optional[float], Dict[str, Scalar], EvaluateResultsAndFailures]]:
        # By default the checkpointing works off of the aggregated evaluation loss from each of the clients
        # NOTE: parameter aggregation occurs **before** evaluation, so the parameters held by the server have been
        # updated prior to this function being called.
        eval_round_results = super().evaluate_round(server_round, timeout)
        if eval_round_results:
            loss_aggregated, metrics_aggregated, (results, failures) = eval_round_results
            if loss_aggregated:
                self._maybe_checkpoint(loss_aggregated, server_round)

        return eval_round_results


ExchangerType = TypeVar("ExchangerType", bound=ParameterExchanger)


class FlServerWithCheckpointing(FlServer, Generic[ExchangerType]):
    def __init__(
        self,
        client_manager: ClientManager,
        model: nn.Module,
        parameter_exchanger: ExchangerType,
        wandb_reporter: Optional[ServerWandBReporter] = None,
        strategy: Optional[Strategy] = None,
        checkpointer: Optional[TorchCheckpointer] = None,
    ) -> None:
        """
        This is a standard FL server but equipped with the assumption that the parameter exchanger is capable of
        hydrating the provided server model fully such that it can be checkpointed. For custom checkpointing
        functionality, one need only override _hydrate_model_for_checkpointing.

        Args:
            client_manager (ClientManager): Determines the mechanism by which clients are sampled by the server, if
                they are to be sampled at all.
            model (nn.Module): This is the torch model to be hydrated by the _hydrate_model_for_checkpointing function
            parameter_exchanger (ExchangerType): This is the parameter exchanger to be used to hydrate the model.
            strategy (Optional[Strategy], optional): The aggregation strategy to be used by the server to handle
                client updates and other information potentially sent by the participating clients. If None the
                strategy is FedAvg as set by the flwr Server.
            wandb_reporter (Optional[ServerWandBReporter], optional): To be provided if the server is to log
                information and results to a Weights and Biases account. If None is provided, no logging occurs.
                Defaults to None.
            checkpointer (Optional[TorchCheckpointer], optional): To be provided if the server should perform
                server side checkpointing based on some criteria. If none, then no server-side checkpointing is
                performed. Defaults to None.
        """
        super().__init__(client_manager, strategy, wandb_reporter, checkpointer)
        self.server_model = model
        # To facilitate model rehydration from server-side state for checkpointing
        self.parameter_exchanger = parameter_exchanger

    def _hydrate_model_for_checkpointing(self) -> nn.Module:
        model_ndarrays = parameters_to_ndarrays(self.parameters)
        self.parameter_exchanger.pull_parameters(model_ndarrays, self.server_model)
        return self.server_model
