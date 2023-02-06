"""
This strategy implements the Federated Learning with client-level DP approach discussed in Differentially Private
Learning with Adaptive Clipping. This function provides a noised version of unweighted FedAvgM.
NOTE: It assumes that the models are packaging clipping bits along with the model parameters. If adaptive clipping is
false, these bits will simply be 0.

Paper: https://arxiv.org/abs/1905.03871
"""

import math
from logging import INFO, WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from flwr.common import (
    EvaluateIns,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarray_to_bytes,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from fl4health.client_managers.base_sampling_manager import BaseSamplingManager
from fl4health.strategies.fedavg_sampling import FedAvgSampling
from fl4health.strategies.noisy_aggregate import (
    gaussian_noisy_aggregate_clipping_bits,
    gaussian_noisy_unweighted_aggregate,
    gaussian_noisy_weighted_aggregate,
)


class ClientLevelDPFedAvgM(FedAvgSampling):
    """
    Performs Federated Averaging with Momentum while performing the required server side update noising required
    for client level differential privacy. If enabled, it performs adaptive clipping rather than fixed threshold
    clipping.
    See Differentially Private Learning with Adaptive Clipping
    """

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        weighted_averaging: bool = False,
        per_client_example_cap: Optional[float] = None,
        adaptive_clipping: bool = False,
        server_learning_rate: float = 1.0,
        clipping_learning_rate: float = 1.0,
        clipping_quantile: float = 0.5,
        initial_clipping_bound: float = 0.1,
        weight_noise_multiplier: float = 1.0,
        clipping_noise_mutliplier: float = 1.0,
        beta: float = 0.9,
    ) -> None:
        """Client-level differentially private federated averaging with momentum and adaptive clipping.

        Implementation based on https://arxiv.org/abs/1905.03871

        Parameters
        ----------
        fraction_fit : float, optional
            Fraction of clients used during training. Defaults to 1.0.
        fraction_evaluate : float, optional
            Fraction of clients used during validation. Defaults to 1.0.
        min_available_clients : int, optional
            Minimum number of total clients in the system. Defaults to 2.
        evaluate_fn : Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]]
            ]
        ]
            Optional function used for validation. Defaults to None.
        on_fit_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure training. Defaults to None.
        on_evaluate_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure validation. Defaults to None.
        accept_failures : bool, optional
            Whether or not accept rounds containing failures. Defaults to True.
        initial_parameters : Parameters
            Initial global model parameters,
            NOTE: we assume that they are not None in this implementation.
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn]
            Metrics aggregation function, optional.
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn]
            Metrics aggregation function, optional.
        weighted_averaging: bool Defaults to False
            Determines whether the FedAvg update is weighted by client dataset size or unweighted
        per_client_example_cap: Optional[float]. Defaults to None.
            The maximum number samples per client. hat{w} in https://arxiv.org/pdf/1710.06963.pdf.
        adaptive_clipping: bool Defaults to False.
            Determines whether adaptive clipping is used in the client DP clipping. If enabled, the model expects the
            last entry of the parameter list to be a binary value indicating whether or not the batch gradient was
            clipped
        server_learning_rate: float Defaults to 1.0
            Learning rate for the server side updates
        clipping_learning_rate: float Defaults to 1.0,
            Learning rate for the clipping bound. Only used if adaptive clipping is turned on
        clipping_quantile: float Defaults to 0.5,
            Quantile we are trying to estimate in adaptive clipping. i.e. P(||g|| < C_t) \approx clipping_quantile.
            Only used if adaptive clipping is turned on
        initial_clipping_bound: float Defaults to 0.1,
            Initial guess for the clipping bound corresponding to the clipping quantile described above
            NOTE: If Adaptive clipping is turned off, this is the clipping bound through out FL training.
        weight_noise_multiplier: float Defaults to 1.0
            Noise multiplier for the noising of gradients
        clipping_noise_mutliplier
            Noise multiplier for the noising of clipping bits
        beta: float Defaults to 0.9
            Momentum weight for previous weight updates
        """
        assert initial_parameters is not None
        assert 0.0 <= clipping_quantile <= 1.0
        self.current_weights = parameters_to_ndarrays(initial_parameters)
        # Tacking on the initial clipping bound to be sent to the clients
        initial_parameters.tensors.append(ndarray_to_bytes(np.array([initial_clipping_bound])))
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        self.weighted_averaging = weighted_averaging
        # If per_client_example_cap is None, it will be set as the total samples across clients
        self.per_client_example_cap = per_client_example_cap
        self.adaptive_clipping = adaptive_clipping
        self.server_learning_rate = server_learning_rate
        self.clipping_learning_rate = clipping_learning_rate
        self.clipping_quantile = clipping_quantile
        self.clipping_bound = initial_clipping_bound
        self.weight_noise_multiplier = weight_noise_multiplier
        self.clipping_noise_mutliplier = clipping_noise_mutliplier
        self.beta = beta

        self.m_t: Optional[NDArrays] = None

    def __repr__(self) -> str:
        rep = f"ClientLevelDPFedAvgM(accept_failures={self.accept_failures})"
        return rep

    def modify_noise_multiplier(self) -> float:
        # Modifying the noise multiplier as in Algorithm 1 of Differentially Private Learning with Adaptive Clipping
        sqrt_argument = pow(self.weight_noise_multiplier, -2.0) - pow(2.0 * self.clipping_noise_mutliplier, -2.0)
        if sqrt_argument < 0.0:
            raise ValueError(
                "Noise Multiplier modification will fail. The relationship of the weight and clipping noise "
                f"multipliers leads to negative sqrt argument {sqrt_argument}"
            )
        return pow(sqrt_argument, -0.5)

    def split_model_weights_and_clipping_bits(
        self, weight_results: List[Tuple[NDArrays, int]]
    ) -> Tuple[List[Tuple[NDArrays, int]], NDArrays]:
        # Clipping bits are packed with the model weights as the last entry in the NDArrays list. We split model
        # weights from these and return both
        client_clipping_bits = []
        client_model_weights = []
        for client_weights, client_n_datapoints in weight_results:
            client_model_weights.append((client_weights[:-1], client_n_datapoints))
            client_clipping_bits.append(client_weights[-1])

        return client_model_weights, client_clipping_bits

    def calculate_update_with_momentum(self, weights_update: NDArrays) -> None:
        if not self.m_t:
            self.m_t = weights_update
        else:
            self.m_t = [
                # NOTE: This is not normalized (beta vs. 1-beta) as used in the original implementation
                self.beta * prev_layer_update + noised_layer_update
                for prev_layer_update, noised_layer_update in zip(self.m_t, weights_update)
            ]

    def update_current_weights(self) -> None:
        assert self.m_t is not None
        self.current_weights = [
            current_layer_weight + self.server_learning_rate * layer_mt
            for current_layer_weight, layer_mt in zip(self.current_weights, self.m_t)
        ]

    def _update_clipping_bound_with_noised_bits(
        self,
        noised_clipping_bits: float,
    ) -> None:

        self.clipping_bound = self.clipping_bound * math.exp(
            -self.clipping_learning_rate * (noised_clipping_bits - self.clipping_quantile)
        )

    def update_clipping_bound(self, clipping_bits: NDArrays) -> None:
        noised_clipping_bits_sum = gaussian_noisy_aggregate_clipping_bits(
            clipping_bits, self.clipping_noise_mutliplier
        )
        self._update_clipping_bound_with_noised_bits(noised_clipping_bits_sum)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate fit using averaging of weights (can be unweighted or weighted) and inject noise and optionally
        perform adaptive clipping updates.
        NOTE: This assumes that the model weights sent back by the clients are UPDATES rather than raw weights. That is
        they are theta_client - theta_server rather than just theta_client
        NOTE: this function packs the clipping bound for clients as the last member of the parameters list
        """

        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # If first round compute total expected client weight and return empty update
        if self.weighted_averaging and server_round == 1:
            successful_client_example_counts = [fit_res.num_examples for _, fit_res in results]
            valid_failures = [fail for fail in failures if not isinstance(fail, BaseException)]
            failed_client_example_counts = [fit_res.num_examples for _, fit_res in valid_failures]
            client_example_counts = successful_client_example_counts + failed_client_example_counts
            total_successful_samples = sum(client_example_counts)
            avg_samples = total_successful_samples / len(client_example_counts)

            # Total number of clients is length of successes plus length of failures
            n_clients = len(client_example_counts)

            # To estimate total_samples we scale avg_samples by the n_total_clients by average samples
            total_samples = avg_samples * n_clients

            self.per_client_example_cap = (
                total_samples if self.per_client_example_cap is None else self.per_client_example_cap
            )

            self.total_client_weight = sum(
                [num_examples / self.per_client_example_cap for num_examples in client_example_counts]
            )

            log(
                INFO,
                """First round reserved for solely fetching client sample counts when weighted_averaging is True.
                No updates to aggregate.""",
            )

            return None, {}

        # Convert results
        weights_updates = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples) for _, fit_res in results
        ]

        weights_updates, clipping_bits = self.split_model_weights_and_clipping_bits(weights_updates)

        noise_multiplier = self.weight_noise_multiplier
        if self.adaptive_clipping:
            # The noise multiplier need only be modified in the event of using adaptive clipping to account for the
            # extra gradient information used to adapt the clipping threshold.
            noise_multiplier = self.modify_noise_multiplier()
            self.update_clipping_bound(clipping_bits)
            log(INFO, f"New Clipping Bound is: {self.clipping_bound}")

        if self.weighted_averaging:
            assert self.per_client_example_cap is not None
            noised_aggregated_update = gaussian_noisy_weighted_aggregate(
                weights_updates,
                noise_multiplier,
                self.clipping_bound,
                self.fraction_fit,
                self.per_client_example_cap,
                self.total_client_weight,
            )
        else:
            noised_aggregated_update = gaussian_noisy_unweighted_aggregate(
                weights_updates,
                noise_multiplier,
                self.clipping_bound,
            )

        # momentum calculation
        self.calculate_update_with_momentum(noised_aggregated_update)
        self.update_current_weights()

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return ndarrays_to_parameters(self.current_weights + [np.array([self.clipping_bound])]), metrics_aggregated

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:

        # This strategy requires the client manager to be of type at least BaseSamplingManager
        assert isinstance(client_manager, BaseSamplingManager)
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)

        config["training"] = (self.weighted_averaging and server_round == 1) is False
        fit_ins = FitIns(parameters, config)

        # Sample clients
        if self.weighted_averaging and server_round == 1:
            clients = client_manager.sample_all(self.min_available_clients)
        else:
            clients = client_manager.sample_fraction(self.fraction_fit, self.min_available_clients)

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""

        # This strategy requires the client manager to be of type at least BaseSamplingManager
        assert isinstance(client_manager, BaseSamplingManager)

        # Do not configure federated evaluation if fraction eval is 0 or server is not initialized
        if self.fraction_evaluate == 0.0 or (self.weighted_averaging and server_round == 1):
            return []

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        clients = client_manager.sample_fraction(self.fraction_evaluate, self.min_available_clients)

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]
