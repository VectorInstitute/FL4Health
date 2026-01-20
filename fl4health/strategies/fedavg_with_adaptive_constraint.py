from collections.abc import Callable
from logging import INFO, WARNING

import numpy as np
from flwr.common import MetricsAggregationFn, NDArrays, Parameters, ndarrays_to_parameters
from flwr.common.logger import log
from flwr.common.typing import FitRes, Scalar
from flwr.server.client_proxy import ClientProxy

from fl4health.parameter_exchange.parameter_packer import ParameterPackerAdaptiveConstraint
from fl4health.strategies.aggregate_utils import aggregate_losses, aggregate_results
from fl4health.strategies.basic_fedavg import BasicFedAvg
from fl4health.utils.functions import decode_and_pseudo_sort_results


class FedAvgWithAdaptiveConstraint(BasicFedAvg):
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: (
            Callable[[int, NDArrays, dict[str, Scalar]], tuple[float, dict[str, Scalar]] | None] | None
        ) = None,
        on_fit_config_fn: Callable[[int], dict[str, Scalar]] | None = None,
        on_evaluate_config_fn: Callable[[int], dict[str, Scalar]] | None = None,
        accept_failures: bool = True,
        initial_parameters: Parameters | None,
        fit_metrics_aggregation_fn: MetricsAggregationFn | None = None,
        evaluate_metrics_aggregation_fn: MetricsAggregationFn | None = None,
        initial_loss_weight: float = 1.0,
        adapt_loss_weight: bool = False,
        loss_weight_delta: float = 0.1,
        loss_weight_patience: int = 5,
        weighted_aggregation: bool = True,
        weighted_eval_losses: bool = True,
        weighted_train_losses: bool = False,
    ) -> None:
        """
        A generalization of the fedavg strategy for approaches that use a penalty constraint that we might want to
        adapt based on the loss trajectory. A quintessential example is FedProx, which uses an \\(\\ell^2\\): penalty
        on model weight drift and potentially adapts the coefficient based on the aggregated loss. In addition to the
        model weights, the server also receives the training loss from the clients. If adaptation is enabled, these
        losses are used to update the loss weight parameter according to the FedProx paper recommendations.

        **NOTE**: Initial parameters are **NOT** optional. They must be passed for this strategy.

        The aggregation strategy for weights is the same as in FedAvg.

        Implementation based on https://arxiv.org/abs/1602.05629.

        Args:
            fraction_fit (float, optional): Fraction of clients used during training. Defaults to 1.0.
            fraction_evaluate (float, optional): Fraction of clients used during validation. Defaults to 1.0.
            min_fit_clients (int, optional): Minimum number of clients used during training. Defaults to 2.
            min_evaluate_clients (int, optional): Minimum number of clients used during validation. Defaults to 2.
            min_available_clients (int, optional): Minimum number of total clients in the system.
                Defaults to 2.
            evaluate_fn (Callable[[int, NDArrays, dict[str, Scalar]], tuple[float, dict[str, Scalar]] | None] | None):
                Optional function used for central server-side evaluation. Defaults to None.
            on_fit_config_fn (Callable[[int], dict[str, Scalar]] | None, optional): Function used to configure
                training by providing a configuration dictionary. Defaults to None.
            on_evaluate_config_fn (Callable[[int], dict[str, Scalar]] | None, optional):
                Function used to configure client-side validation by providing a ``Config`` dictionary.
                Defaults to None.
            accept_failures (bool, optional): Whether or not accept rounds containing failures. Defaults to True.
            initial_parameters (Parameters | None, optional): Initial global model parameters.
            fit_metrics_aggregation_fn (MetricsAggregationFn | None, optional): Metrics aggregation function.
                Defaults to None.
            evaluate_metrics_aggregation_fn (MetricsAggregationFn | None, optional): Metrics aggregation function.
                Defaults to None.
            initial_loss_weight (float): Initial loss weight (mu in FedProx). If adaptivity is false, then this is the
                constant weight used for all clients.
            adapt_loss_weight (bool, optional): Determines whether the value of mu is adaptively modified by
                the server based on aggregated train loss. Defaults to False.
            loss_weight_delta (float, optional): This is the amount by which the server changes the value of mu
                based on the modification criteria. Only applicable if adaptivity is on. Defaults to 0.1.
            loss_weight_patience (int, optional): This is the number of rounds a server must see decreasing
                aggregated train loss before reducing the value of mu. Only applicable if adaptivity is on.
                Defaults to 5.
            weighted_aggregation (bool, optional): Determines whether parameter aggregation is a linearly weighted
                average or a uniform average. FedAvg default is weighted average by client dataset counts.
                Defaults to True.
            weighted_eval_losses (bool, optional): Determines whether losses during evaluation are linearly weighted
                averages or a uniform average. FedAvg default is weighted average of the losses by client dataset
                counts. Defaults to True.
            weighted_train_losses (bool, optional): Determines whether the training losses from the clients should be
                aggregated using a weighted or unweighted average. These aggregated losses are used to adjust the
                proximal weight in the adaptive setting. Defaults to False.
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
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
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
            weighted_aggregation=weighted_aggregation,
            weighted_eval_losses=weighted_eval_losses,
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
        Aggregate the results from the federated fit round and, if applicable, determine whether the constraint weight
        should be updated based on the aggregated loss seen on the clients.

        Args:
            server_round (int): Indicates the server round we're currently on.
            results (list[tuple[ClientProxy, FitRes]]): The client identifiers and the results of their local training
                that need to be aggregated on the server-side. For adaptive constraints, the clients pack the weights
                to be aggregated along with the training loss seen during their local training cycle.
            failures (list[tuple[ClientProxy, FitRes] | BaseException]): These are the results and exceptions
                from clients that experienced an issue during training, such as timeouts or exceptions.

        Returns:
            (tuple[Parameters | None, dict[str, Scalar]]): The aggregated model weights and the metrics dictionary.
                For adaptive constraints, the server also packs a constraint weight to be sent to the clients. This is
                sent even if adaptive constraint weights are turned off and the value simply remains constant.
        """
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Sorting the results by elements and sample counts. This is primarily to reduce numerical fluctuations in
        # summing the numpy arrays during aggregation. This ensures that addition will occur in the same order,
        # reducing numerical fluctuation.
        decoded_and_sorted_results = [
            (weights, sample_counts) for _, weights, sample_counts in decode_and_pseudo_sort_results(results)
        ]

        # Convert results with packed params of model weights and training loss
        weights_and_counts: list[tuple[NDArrays, int]] = []
        train_losses_and_counts: list[tuple[int, float]] = []
        for weights, sample_count in decoded_and_sorted_results:
            updated_weights, train_loss = self.parameter_packer.unpack_parameters(weights)
            weights_and_counts.append((updated_weights, sample_count))
            train_losses_and_counts.append((sample_count, train_loss))

        # Aggregate them in a weighted or unweighted fashion based on settings.
        weights_aggregated = aggregate_results(weights_and_counts, self.weighted_aggregation)

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

        parameters = self.parameter_packer.pack_parameters(weights_aggregated, self.loss_weight)
        return ndarrays_to_parameters(parameters), metrics_aggregated

    def _maybe_update_constraint_weight_param(self, loss: float) -> None:
        """
        Update constraint weight parameter if ``adaptive_loss_weight`` is set to True. Regardless of whether adaptivity
        is turned on at this time, the previous loss seen by the server is updated.

        **NOTE**: For adaptive constraint losses, including FedProx, this loss is exchanged (along with the
        weights) by each client and is the **VANILLA** loss that does not include the additional penalty losses.

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
