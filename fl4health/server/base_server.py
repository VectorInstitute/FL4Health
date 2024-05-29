import datetime
from logging import DEBUG, INFO, WARNING
from typing import Dict, Generic, List, Optional, Tuple, TypeVar, Union

import torch.nn as nn
from flwr.common import EvaluateRes, Parameters
from flwr.common.logger import log
from flwr.common.parameter import parameters_to_ndarrays
from flwr.common.typing import Scalar
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.history import History
from flwr.server.server import EvaluateResultsAndFailures, FitResultsAndFailures, Server, evaluate_clients
from flwr.server.strategy import Strategy

from fl4health.checkpointing.checkpointer import TorchCheckpointer
from fl4health.parameter_exchange.parameter_exchanger_base import ParameterExchanger
from fl4health.reporting.fl_wandb import ServerWandBReporter
from fl4health.reporting.metrics import MetricsReporter
from fl4health.utils.metrics import TestMetricPrefix
from fl4health.server.polling import poll_clients
from fl4health.strategies.strategy_with_poll import StrategyWithPolling


class FlServer(Server):
    def __init__(
        self,
        client_manager: ClientManager,
        strategy: Optional[Strategy] = None,
        wandb_reporter: Optional[ServerWandBReporter] = None,
        checkpointer: Optional[TorchCheckpointer] = None,
        metrics_reporter: Optional[MetricsReporter] = None,
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
            metrics_reporter (Optional[MetricsReporter], optional): A metrics reporter instance to record the metrics
                during the execution. Defaults to an instance of MetricsReporter with default init parameters.
        """

        super().__init__(client_manager=client_manager, strategy=strategy)
        self.wandb_reporter = wandb_reporter
        self.checkpointer = checkpointer

        if metrics_reporter is not None:
            self.metrics_reporter = metrics_reporter
        else:
            self.metrics_reporter = MetricsReporter()

    def fit(self, num_rounds: int, timeout: Optional[float]) -> History:
        self.metrics_reporter.add_to_metrics({"type": "server", "fit_start": datetime.datetime.now()})

        history = super().fit(num_rounds, timeout)
        if self.wandb_reporter:
            # report history to W and B
            self.wandb_reporter.report_metrics(num_rounds, history)

        self.metrics_reporter.add_to_metrics(
            data={
                "fit_end": datetime.datetime.now(),
                "metrics_centralized": history.metrics_centralized,
                "losses_centralized": history.losses_centralized,
            }
        )

        return history

    def fit_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]]:
        self.metrics_reporter.add_to_metrics_at_round(server_round, data={"fit_start": datetime.datetime.now()})

        fit_round_results = super().fit_round(server_round, timeout)

        if fit_round_results is not None:
            _, metrics_aggregated, _ = fit_round_results
            self.metrics_reporter.add_to_metrics_at_round(
                server_round,
                data={
                    "metrics_aggregated": metrics_aggregated,
                    "fit_end": datetime.datetime.now(),
                },
            )

        return fit_round_results

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

    def _maybe_checkpoint(
        self, loss_aggregated: float, metrics_aggregated: Dict[str, Scalar], server_round: int
    ) -> None:
        if self.checkpointer:
            try:
                model = self._hydrate_model_for_checkpointing()
                self.checkpointer.maybe_checkpoint(model, loss_aggregated, metrics_aggregated)
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

    def _unpack_metrics(
        self, results: List[Tuple[ClientProxy, EvaluateRes]]
    ) -> Tuple[List[Tuple[ClientProxy, EvaluateRes]], List[Tuple[ClientProxy, EvaluateRes]]]:
        val_results = []
        test_results = []

        for client_proxy, eval_res in results:
            val_metrics = {k: v for k, v in eval_res.metrics.items() if not k.startswith(TestMetricPrefix.TEST_PREFIX)}
            test_metrics = {k: v for k, v in eval_res.metrics.items() if k.startswith(TestMetricPrefix.TEST_PREFIX)}

            if len(test_metrics) > 0:
                assert TestMetricPrefix.TEST_PREFIX + "loss" in test_metrics and TestMetricPrefix.TEST_PREFIX + "num_examples" in test_metrics, (
                    TestMetricPrefix.TEST_PREFIX + "loss and " + TestMetricPrefix.TEST_PREFIX + "num_examples keys must be present "
                    "in test_metrics dictionary for aggregation"
                )
                # Remove loss and num_examples from test_metrics if they exist
                test_loss = float(test_metrics.pop(TestMetricPrefix.TEST_PREFIX + "loss"))
                test_num_examples = int(test_metrics.pop(TestMetricPrefix.TEST_PREFIX + "num_examples"))
                test_eval_res = EvaluateRes(eval_res.status, test_loss, test_num_examples, test_metrics)
                test_results.append((client_proxy, test_eval_res))

            val_eval_res = EvaluateRes(eval_res.status, eval_res.loss, eval_res.num_examples, val_metrics)
            val_results.append((client_proxy, val_eval_res))

        return val_results, test_results

    def _handle_result_aggregation(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        val_results, test_results = self._unpack_metrics(results)

        # Aggregate the validation results
        val_aggregated_result: Tuple[
            Optional[float],
            Dict[str, Scalar],
        ] = self.strategy.aggregate_evaluate(server_round, val_results, failures)
        val_loss_aggregated, val_metrics_aggregated = val_aggregated_result

        # Aggregate the test results if they are present
        if len(test_results) > 0:
            test_aggregated_result: Tuple[
                Optional[float],
                Dict[str, Scalar],
            ] = self.strategy.aggregate_evaluate(server_round, test_results, failures)
            test_loss_aggregated, test_metrics_aggregated = test_aggregated_result

            for key, value in test_metrics_aggregated.items():
                val_metrics_aggregated[key] = value
            if test_loss_aggregated is not None:
                val_metrics_aggregated[TestMetricPrefix.TEST_PREFIX + "loss - aggregated"] = test_loss_aggregated

        return val_loss_aggregated, val_metrics_aggregated

    def _evaluate_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[Tuple[Optional[float], Dict[str, Scalar], EvaluateResultsAndFailures]]:
        """Validate current global model on a number of clients."""
        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_evaluate(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self._client_manager,
        )
        if not client_instructions:
            log(INFO, "evaluate_round %s: no clients selected, cancel", server_round)
            return None
        log(
            DEBUG,
            "evaluate_round %s: strategy sampled %s clients (out of %s)",
            server_round,
            len(client_instructions),
            self._client_manager.num_available(),
        )
        # Collect `evaluate` results from all clients participating in this round
        results, failures = evaluate_clients(
            client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
        )
        log(
            DEBUG,
            "evaluate_round %s received %s results and %s failures for Validation",
            server_round,
            len(results),
            len(failures),
        )

        val_loss_aggregated, val_metrics_aggregated = self._handle_result_aggregation(server_round, results, failures)

        return val_loss_aggregated, val_metrics_aggregated, (results, failures)

    def evaluate_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[Tuple[Optional[float], Dict[str, Scalar], EvaluateResultsAndFailures]]:
        self.metrics_reporter.add_to_metrics_at_round(server_round, data={"evaluate_start": datetime.datetime.now()})

        # By default the checkpointing works off of the aggregated evaluation loss from each of the clients
        # NOTE: parameter aggregation occurs **before** evaluation, so the parameters held by the server have been
        # updated prior to this function being called.
        eval_round_results = self._evaluate_round(server_round, timeout)
        if eval_round_results:
            loss_aggregated, metrics_aggregated, _ = eval_round_results
            if loss_aggregated:
                self._maybe_checkpoint(loss_aggregated, metrics_aggregated, server_round)

            self.metrics_reporter.add_to_metrics_at_round(
                server_round,
                data={
                    "metrics_aggregated": metrics_aggregated,
                    "loss_aggregated": loss_aggregated,
                    "evaluate_end": datetime.datetime.now(),
                },
            )

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
        metrics_reporter: Optional[MetricsReporter] = None,
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
        super().__init__(client_manager, strategy, wandb_reporter, checkpointer, metrics_reporter)
        self.server_model = model
        # To facilitate model rehydration from server-side state for checkpointing
        self.parameter_exchanger = parameter_exchanger

    def _hydrate_model_for_checkpointing(self) -> nn.Module:
        model_ndarrays = parameters_to_ndarrays(self.parameters)
        self.parameter_exchanger.pull_parameters(model_ndarrays, self.server_model)
        return self.server_model
