import datetime
from collections.abc import Sequence
from logging import INFO, WARNING
from pathlib import Path

import torch
from flwr.common import EvaluateIns, EvaluateRes, MetricsAggregationFn, Parameters, Scalar
from flwr.common.logger import log
from flwr.common.parameter import ndarrays_to_parameters
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.history import History
from flwr.server.server import EvaluateResultsAndFailures, Server, evaluate_clients

from fl4health.client_managers.base_sampling_manager import BaseFractionSamplingManager
from fl4health.reporting.base_reporter import BaseReporter
from fl4health.utils.random import generate_hash


class EvaluateServer(Server):
    def __init__(
        self,
        client_manager: ClientManager,
        fraction_evaluate: float,
        model_checkpoint_path: Path | None = None,
        evaluate_config: dict[str, Scalar] | None = None,
        evaluate_metrics_aggregation_fn: MetricsAggregationFn | None = None,
        accept_failures: bool = True,
        min_available_clients: int = 1,
        reporters: Sequence[BaseReporter] | None = None,
    ) -> None:
        """
        Args:
            client_manager (ClientManager): Determines the mechanism by which clients are sampled by the server, if
                they are to be sampled at all.
            fraction_evaluate (float): Fraction of clients used during evaluation.
            model_checkpoint_path (Path | None, optional): Server side model checkpoint path to load global model
                from. Defaults to None.
            evaluate_config (dict[str, Scalar] | None, optional): Configuration dictionary to configure evaluation
                on clients. Defaults to None.
            evaluate_metrics_aggregation_fn (MetricsAggregationFn | None, optional):  Metrics aggregation function.
                 Defaults to None.
            accept_failures (bool, optional): Whether or not accept rounds containing failures. Defaults to True.
            min_available_clients (int, optional): Minimum number of total clients in the system. Defaults to 1.
                Defaults to 1.
            reporters (Sequence[BaseReporter], optional): A sequence of FL4Health reporters which the client should
                send data to.
        """
        # We aren't aggregating model weights, so setting the strategy to be none.
        super().__init__(client_manager=client_manager, strategy=None)
        self.model_checkpoint_path = model_checkpoint_path
        # Load model parameters if checkpoint provided, otherwise leave as empty params
        if model_checkpoint_path:
            self.parameters = self.load_model_checkpoint_to_parameters()
        self.fraction_evaluate = fraction_evaluate
        self.evaluate_config = evaluate_config
        self.min_available_clients = min_available_clients
        self.accept_failures = accept_failures
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        if self.fraction_evaluate < 1.0:
            log(
                INFO,
                f"Fraction Evaluate is {self.fraction_evaluate}. "
                "Thus, some clients may not participate in evaluation",
            )
        self.server_name = generate_hash()
        self.reporters = [] if reporters is None else list(reporters)
        for r in self.reporters:
            r.initialize(id=self.server_name)

    def load_model_checkpoint_to_parameters(self) -> Parameters:
        assert self.model_checkpoint_path
        log(INFO, f"Loading model checkpoint at: {self.model_checkpoint_path.__str__()}")
        model = torch.load(self.model_checkpoint_path)
        # Extracting all parameters from the model to be sent to the clients
        parameters = ndarrays_to_parameters([val.cpu().numpy() for _, val in model.state_dict().items()])
        log(INFO, "Model loaded and state converted to parameters")
        return parameters

    def fit(self, num_rounds: int, timeout: float | None) -> tuple[History, float]:
        """
        In order to head off training and only run eval, we have to override the fit function as this is
        essentially the entry point for federated learning from the app.

        Args:
            num_rounds (int): Not used.
            timeout (float | None): Timeout in seconds that the server should wait for the clients to respond.
                If none, then it will wait for the minimum number to respond indefinitely.

        Returns:
            tuple[History, float]: The first element of the tuple is a History object containing the aggregated
                metrics returned from the clients. Tuple also contains elapsed time in seconds for round.
        """
        history = History()

        # Run Federated Evaluation
        log(INFO, "Federated Evaluation Starting")
        start_time = datetime.datetime.now()

        for reporter in self.reporters:
            reporter.report(
                {
                    "fit_start": str(start_time),
                    "host_type": "server",
                }
            )
        # We're only performing federated evaluation. So we make use of the evaluate round function, but simply
        # perform such evaluation once.
        res_fed = self.federated_evaluate(timeout=timeout)
        end_time = datetime.datetime.now()

        for r in self.reporters:
            r.report(
                {
                    "fit_elapsed_time": str(start_time - end_time),
                    "fit_end": str(end_time),
                    "num_rounds": num_rounds,
                    "host_type": "server",
                }
            )
        if res_fed:
            _, evaluate_metrics_fed, _ = res_fed
            if evaluate_metrics_fed:
                history.add_metrics_distributed(server_round=0, metrics=evaluate_metrics_fed)
                if evaluate_metrics_fed:
                    for r in self.reporters:
                        r.report({"fit_metrics": evaluate_metrics_fed})

        # Bookkeeping
        elapsed = end_time - start_time
        log(INFO, "Federated Evaluation Finished in %s", str(elapsed))
        return history, elapsed.total_seconds()

    def federated_evaluate(
        self,
        timeout: float | None,
    ) -> tuple[float | None, dict[str, Scalar], EvaluateResultsAndFailures] | None:
        """
        Validate current global model on a number of clients.

        Args:
            timeout (float | None): Timeout in seconds that the server should wait for the clients to response.
                If none, then it will wait for the minimum number to respond indefinitely.

        Returns:
            tuple[float | None, dict[str, Scalar], EvaluateResultsAndFailures] | None: The first value is the
                loss, which is ignored since we pack loss from the global and local models into the metrics dictionary
                The second is the aggregated metrics passed from the clients, the third is the set of raw results and
                failure objects returned by the clients.
        """

        # Get clients and their respective instructions from client manager
        client_instructions = self.configure_evaluate()

        if not client_instructions:
            log(INFO, "Federated Evaluation: no clients selected, cancel")
            return None
        log(
            INFO,
            f"Federated Evaluation: Client manager sampled {len(client_instructions)} "
            f"clients (out of {self._client_manager.num_available()})",
        )

        # Collect `evaluate` results from all clients participating in this round
        results, failures = evaluate_clients(
            client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
            group_id=0,
        )
        log(
            INFO,
            f"Federated Evaluation received {len(results)} results and {len(failures)} failures",
        )

        # Aggregate the evaluation results, note that we assume that the losses have been packed and aggregated with
        # the metrics. A dummy loss is returned by each of the clients. We therefore return none for the aggregated
        # loss
        aggregated_result: tuple[
            float | None,
            dict[str, Scalar],
        ] = self.aggregate_evaluate(results, failures)

        _, metrics_aggregated = aggregated_result
        return None, metrics_aggregated, (results, failures)

    def configure_evaluate(self) -> list[tuple[ClientProxy, EvaluateIns]]:
        """
        Configure the next round of evaluation. This handles the two different was that a set of clients might be
        sampled.

        Returns:
            list[tuple[ClientProxy, EvaluateIns]]: List of configuration instructions for the clients selected by the
                client manager for evaluation. These configuration objects are sent to the clients to customize
                evaluation.
        """
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # Parameters and config
        config = {}
        if self.evaluate_config is not None:
            # Custom evaluation config function provided
            config = self.evaluate_config
        evaluate_ins = EvaluateIns(self.parameters, config)

        # Sample clients
        if isinstance(self._client_manager, BaseFractionSamplingManager):
            clients = self._client_manager.sample_fraction(self.fraction_evaluate, self.min_available_clients)
        else:
            sample_size = int(self._client_manager.num_available() * self.fraction_evaluate)
            clients = self._client_manager.sample(num_clients=sample_size, min_num_clients=self.min_available_clients)

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_evaluate(
        self,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[tuple[ClientProxy, EvaluateRes] | BaseException],
    ) -> tuple[float | None, dict[str, Scalar]]:
        """
        Aggregate evaluation results using the evaluate_metrics_aggregation_fn provided. Note that a dummy loss is
        returned as we assume that it was packed into the metrics dictionary for this functionality.

        Args:
            results (list[tuple[ClientProxy, EvaluateRes]]): List of results objects that have the metrics returned
                from each client, if successful, along with the number of samples used in the evaluation.
            failures (list[tuple[ClientProxy, EvaluateRes] | BaseException]): Failures reported by the clients
                along with the client id, the results that we passed, if any, and the associated exception if one was
                raised.

        Returns:
            tuple[float | None, dict[str, Scalar]]: A dummy float for the "loss" (these are packed with the metrics)
                and the aggregated metrics dictionary.
        """
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        else:
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        # Losses contained in results are dummy values for federated evaluation. It is assume that the client losses
        # are packed, and therefore aggregated, in the metrics dictionary.
        return None, metrics_aggregated
