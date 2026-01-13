from collections.abc import Callable
from logging import INFO, WARNING

import numpy as np
from flwr.common import MetricsAggregationFn, NDArrays, Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.logger import log
from flwr.common.typing import FitRes, Scalar
from flwr.server.client_proxy import ClientProxy

from fl4health.parameter_exchange.parameter_packer import ParameterPackerAdaptiveConstraint
from fl4health.strategies.aggregate_utils import aggregate_losses
from fl4health.strategies.feddg_ga import FairnessMetric, FedDgGa


class FedDgGaAdaptiveConstraint(FedDgGa):
    def __init__(
        self,
        *,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: (
            Callable[[int, NDArrays, dict[str, Scalar]], tuple[float, dict[str, Scalar]] | None] | None
        ) = None,
        on_fit_config_fn: Callable[[int], dict[str, Scalar]] | None = None,
        on_evaluate_config_fn: Callable[[int], dict[str, Scalar]] | None = None,
        accept_failures: bool = True,
        initial_parameters: Parameters,
        fit_metrics_aggregation_fn: MetricsAggregationFn | None = None,
        evaluate_metrics_aggregation_fn: MetricsAggregationFn | None = None,
        initial_loss_weight: float = 1.0,
        adapt_loss_weight: bool = False,
        loss_weight_delta: float = 0.1,
        loss_weight_patience: int = 5,
        weighted_train_losses: bool = False,
        fairness_metric: FairnessMetric | None = None,
        adjustment_weight_step_size: float = 0.2,
    ):
        """
        Strategy for the FedDG-GA algorithm (Federated Domain Generalization with Generalization Adjustment,
        Zhang et al. 2023) combined with the Adaptive Strategy for Auxiliary constraints like FedProx. See
        documentation on ``FedAvgWithAdaptiveConstraint`` for more information.

        **NOTE**: Initial parameters are **NOT** optional. They must be passed for this strategy.

        Args:
            min_fit_clients (int, optional): Minimum number of clients used during training. Defaults to 2.
            min_evaluate_clients (int, optional): Minimum number of clients used during validation. Defaults to 2.
            min_available_clients (int, optional): Minimum number of total clients in the system. Defaults to 2.
            evaluate_fn (Callable[[int, NDArrays, dict[str, Scalar]], tuple[float, dict[str, Scalar]] | None] | None):
                Optional function used for validation. Defaults to None.
            on_fit_config_fn (Callable[[int], dict[str, Scalar]] | None, optional): Function used to configure
                training. Defaults to None.
            on_evaluate_config_fn (Callable[[int], dict[str, Scalar]] | None, optional): Function used to configure
                validation. Defaults to None
            initial_parameters (Parameters): Initial global model parameters.
            accept_failures (bool, optional): Whether or not accept rounds containing failures. Defaults to True.
            fit_metrics_aggregation_fn (MetricsAggregationFn | None, optional):
                Metrics aggregation function, Defaults to None.
            evaluate_metrics_aggregation_fn (MetricsAggregationFn | None, optional):
                Metrics aggregation function. Defaults to None.
            initial_loss_weight (float, optional): Initial penalty loss weight (mu in FedProx). If adaptivity is false,
                then this is the constant weight used for all clients. Defaults to 1.0.
            adapt_loss_weight (bool, optional): Determines whether the value of the penalty loss weight is adaptively
                modified by the server based on aggregated train loss. Defaults to False.
            loss_weight_delta (float, optional): This is the amount by which the server changes the value of the
                penalty loss weight based on the modification criteria. Only applicable if adaptivity is on.
                Defaults to 0.1.
            loss_weight_patience (int, optional): This is the number of rounds a server must see decreasing
                aggregated train loss before reducing the value of the penalty loss weight. Only applicable if
                adaptivity is on. Defaults to 5.
            weighted_train_losses (bool, optional): Determines whether the training losses from the clients should be
                aggregated using a weighted or unweighted average. These aggregated losses are used to adjust the
                proximal weight in the adaptive setting. Defaults to False.
            fairness_metric (FairnessMetric | None, optional): he metric to evaluate the local model of each
                client against the global model in order to determine their adjustment weight for aggregation.
                Can be set to any default metric in ``FairnessMetricType`` or set to use a custom metric.
                Optional, default is ``FairnessMetric(FairnessMetricType.LOSS)`` when specified as None.
            adjustment_weight_step_size (float, optional): The step size to determine the magnitude of change for
                the generalization adjustment weight. It has to be ``0 < adjustment_weight_step_size < 1.``
                Optional, default is 0.2.
        """
        self.loss_weight = initial_loss_weight
        self.adapt_loss_weight = adapt_loss_weight

        if self.adapt_loss_weight:
            self.loss_weight_delta = loss_weight_delta
            self.loss_weight_patience = loss_weight_patience
            self.loss_weight_patience_counter: int = 0

        self.previous_loss = float("inf")

        if initial_parameters:
            self.add_auxiliary_information(initial_parameters)

        super().__init__(
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            fairness_metric=fairness_metric,
            adjustment_weight_step_size=adjustment_weight_step_size,
        )

        self.parameter_packer = ParameterPackerAdaptiveConstraint()
        self.weighted_train_losses = weighted_train_losses

    def add_auxiliary_information(self, original_parameters: Parameters) -> None:
        """
        Function for adding in the ``loss_weight`` to the provided set of parameters. This function is meant to be
        called after a server requests model weight initialization from a client, allowing the proper information to
        be included with the model parameters when sent to all clients for model initialization etc.

        Args:
            original_parameters (Parameters): Original set of parameters provided by a client for model weight
                initialization
        """
        original_parameters.tensors.extend(ndarrays_to_parameters([np.array(self.loss_weight)]).tensors)

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        """
        Aggregate fit results by weighing them against the adjustment weights and then summing them.

        Collects the fit metrics that will be used to change the adjustment weights for the next round.

        If applicable, determine whether the constraint weight should be updated based on the aggregated loss
        seen on the clients.

        Args:
            server_round: (int) The current server round.
            results: (list[tuple[ClientProxy, FitRes]]) The clients' fit results.
            failures: (list[tuple[ClientProxy, FitRes] | BaseException]) The clients' fit failures.

        Returns:
            (tuple[Parameters | None, dict[str, Scalar]]) A tuple containing the aggregated parameters
                and the aggregated fit metrics. For adaptive constraints, the server also packs a constraint weight
                to be sent to the clients. This is sent even if adaptive constraint weights are turned off and
                the value simply remains constant.
        """
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results with packed params of model weights and training loss. The results list is modified in-place
        # to only contain model parameters for use in the Fed-DGGA calculations and aggregation
        train_losses_and_counts = self._unpack_weights_and_losses(results)

        # Aggregate train loss
        train_losses_aggregated = aggregate_losses(train_losses_and_counts, self.weighted_train_losses)
        self._maybe_update_constraint_weight_param(train_losses_aggregated)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        self.train_metrics = {}
        for client_proxy, fit_res in results:
            self.train_metrics[client_proxy.cid] = fit_res.metrics

        weights_aggregated = self.weight_and_aggregate_results(results)

        parameters = self.parameter_packer.pack_parameters(weights_aggregated, self.loss_weight)
        return ndarrays_to_parameters(parameters), metrics_aggregated

    def _unpack_weights_and_losses(self, results: list[tuple[ClientProxy, FitRes]]) -> list[tuple[int, float]]:
        """
        This function takes results returned from a fit round from each of the participating clients and unpacks the
        information into the appropriate objects. The parameters contained in the FitRes object are unpacked to
        separate the model weights from the training losses. The model weights are reinserted into the parameters
        of the FitRes objects and the losses (along with sample counts) are placed in a list and returned.

        **NOTE**: The results that are passed to this function are **MODIFIED IN-PLACE**.

        Args:
            results (list[tuple[ClientProxy, FitRes]]): The results produced in a fitting round by each of the clients
                these the FitRes object contains both model weights and training losses which need to be processed.

        Returns:
            (list[tuple[int, float]]): A list of the training losses produced by client training
        """
        train_losses_and_counts: list[tuple[int, float]] = []
        for _, fit_res in results:
            sample_count = fit_res.num_examples
            updated_weights, train_loss = self.parameter_packer.unpack_parameters(
                parameters_to_ndarrays(fit_res.parameters)
            )
            # Modify the parameters in-place to just be the model weights.
            fit_res.parameters = ndarrays_to_parameters(updated_weights)
            train_losses_and_counts.append((sample_count, train_loss))

        return train_losses_and_counts

    def _maybe_update_constraint_weight_param(self, loss: float) -> None:
        """
        Update constraint weight parameter if ``adaptive_loss_weight`` is set to True. Regardless of whether adaptivity
        is turned on at this time, the previous loss seen by the server is updated.

        **NOTE**: For adaptive constraint losses, including FedProx, this loss is exchanged (along with the
        weights) by each client and is the VANILLA loss that does not include the additional penalty losses.

        Args:
            loss (float): This is the loss to which we compare the previous loss seen by the server. For Adaptive
                Constraint clients this should be the aggregated training loss seen by each client participating in
                training.
        """
        if self.adapt_loss_weight:
            if loss <= self.previous_loss:
                self.loss_weight_patience_counter += 1
                if self.loss_weight_patience_counter == self.loss_weight_patience:
                    self.loss_weight -= self.loss_weight_delta
                    self.loss_weight = max(0.0, self.loss_weight)
                    self.loss_weight_patience_counter = 0
                    log(INFO, f"Aggregate training loss has dropped {self.loss_weight_patience} rounds in a row")
                    log(INFO, f"Constraint weight is decreased to {self.loss_weight}")
            else:
                self.loss_weight += self.loss_weight_delta
                self.loss_weight_patience_counter = 0
                log(
                    INFO,
                    f"Aggregate training loss increased this round: Current loss {loss}, "
                    f"Previous loss: {self.previous_loss}",
                )
                log(INFO, f"Constraint weight is increased by {self.loss_weight_delta} to {self.loss_weight}")
        self.previous_loss = loss
