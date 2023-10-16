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

from fl4health.client_managers.base_sampling_manager import BaseFractionSamplingManager
from fl4health.parameter_exchange.parameter_packer import ParameterPackerWithClippingBit
from fl4health.strategies.basic_fedavg import BasicFedAvg
from fl4health.strategies.noisy_aggregate import (
    gaussian_noisy_aggregate_clipping_bits,
    gaussian_noisy_unweighted_aggregate,
    gaussian_noisy_weighted_aggregate,
)


class ClientLevelDPFedAvgM(BasicFedAvg):
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
        weighted_aggregation: bool = False,
        weighted_eval_losses: bool = True,
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
        """
        This strategy implements the Federated Learning with client-level DP approach discussed in
        Differentially Private Learning with Adaptive Clipping. This function provides a noised version of unweighted
        FedAvgM.
        NOTE: It assumes that the models are packaging clipping bits along with the model parameters. If adaptive
        clipping is false, these bits will simply be 0.

        Paper: https://arxiv.org/abs/1905.03871
        If enabled, it performs adaptive clipping rather than fixed threshold clipping.

        Args:
            fraction_fit (float, optional): Fraction of clients used during training. Defaults to 1.0.
            fraction_evaluate (float, optional): Fraction of clients used during validation. Defaults to 1.0.
            min_available_clients (int, optional): Minimum number of clients used during validation.
                Defaults to 2.
            evaluate_fn (Optional[
                Callable[[int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]
            ]):
                Optional function used for central server-side evaluation. Defaults to None.
            on_fit_config_fn (Optional[Callable[[int], Dict[str, Scalar]]], optional):
                Function used to configure training by providing a configuration dictionary. Defaults to None.
            on_evaluate_config_fn (Optional[Callable[[int], Dict[str, Scalar]]], optional):
                Function used to configure server-side central validation by providing a Config dictionary.
               Defaults to None.
            accept_failures (bool, optional): Whether or not accept rounds containing failures. Defaults to True.
            initial_parameters (Optional[Parameters], optional): Initial global model parameters. This strategy assumes
                that the initial parameters is not None. So they need to be set inspite of the optional tag.
            fit_metrics_aggregation_fn (Optional[MetricsAggregationFn], optional): Metrics aggregation function.
                Defaults to None.
            evaluate_metrics_aggregation_fn (Optional[MetricsAggregationFn], optional): Metrics aggregation function.
                Defaults to None.
            weighted_aggregation (bool, optional): Determines whether the FedAvg update is weighted by client dataset
                size or unweighted. Defaults to False.
            weighted_eval_losses (bool, optional): Determines whether losses during evaluation are linearly weighted
                averages or a uniform average. FedAvg default is weighted average of the losses by client dataset
                counts. Defaults to True.
            per_client_example_cap (Optional[float], optional): The maximum number samples per client. hat{w} in
                https://arxiv.org/pdf/1710.06963.pdf. Defaults to None.
            adaptive_clipping (bool, optional): If enabled, the model expects the last entry of the parameter list to
                be a binary value indicating whether or not the batch gradient was clipped. Defaults to False.
            server_learning_rate (float, optional): Learning rate for the server side updates. Defaults to 1.0.
            clipping_learning_rate (float, optional): Learning rate for the clipping bound. Only used if adaptive
                clipping is turned on. Defaults to 1.0.
            clipping_quantile (float, optional): Quantile we are trying to estimate in adaptive clipping.
                i.e. P(||g|| < C_t) \approx clipping_quantile. Only used if adaptive clipping is turned on.
                Defaults to 0.5.
            initial_clipping_bound (float, optional):  Initial guess for the clipping bound corresponding to the
                clipping quantile described above. NOTE: If Adaptive clipping is turned off, this is the clipping
                bound through out FL training.. Defaults to 0.1.
            weight_noise_multiplier (float, optional): Noise multiplier for the noising of gradients. Defaults to 1.0.
            clipping_noise_mutliplier (float, optional): Noise multiplier for the noising of clipping bits.
                Defaults to 1.0.
            beta (float, optional): Momentum weight for previous weight updates. If it is 0, there is no momentum.
                Defaults to 0.9.
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
            weighted_aggregation=weighted_aggregation,
            weighted_eval_losses=weighted_eval_losses,
        )
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

        # Parameter Packer to handle packing and unpacking parameters with clipping bit
        self.parameter_packer = ParameterPackerWithClippingBit()

        # Weighted averaging requires list of sample counts
        # to compute client weights. Set by server after polling clients.
        self.sample_counts: Optional[List[int]] = None
        self.m_t: Optional[NDArrays] = None

    def __repr__(self) -> str:
        rep = f"ClientLevelDPFedAvgM(accept_failures={self.accept_failures})"
        return rep

    def modify_noise_multiplier(self) -> float:
        """
        Modifying the noise multiplier as in Algorithm 1 of Differentially Private Learning with Adaptive Clipping.
        This is done to ensure the privacy accountant computes the correct privacy values.

        Raises:
            ValueError: If the noise multipler and the clipping noise multiplier are not well related then we'll end up
                with a sqrt of a negative number. If this happens a value error is raised.

        Returns:
            float: The modified noise multiplier when performing adaptive clipping.
        """
        # Modifying the noise multiplier as in Algorithm 1 of Differentially Private Learning with Adaptive Clipping
        sqrt_argument = pow(self.weight_noise_multiplier, -2.0) - pow(2.0 * self.clipping_noise_mutliplier, -2.0)
        if sqrt_argument < 0.0:
            raise ValueError(
                "Noise Multiplier modification will fail. The relationship of the weight and clipping noise "
                f"multipliers leads to negative sqrt argument {sqrt_argument}"
            )
        return pow(sqrt_argument, -0.5)

    def split_model_weights_and_clipping_bits(
        self, results: List[Tuple[ClientProxy, FitRes]]
    ) -> Tuple[List[Tuple[NDArrays, int]], NDArrays]:
        """
        Given results from an FL round of training, this function splits the result into sets of
        (weights, training counts) and clipping bits. The split is required because the clipping bits are packed with
        the weights in order to communicate them back to the server. The parameter packer facilitates this splitting.

        Args:
            results (List[Tuple[ClientProxy, FitRes]]): The client identifiers and the results of their local training
                that need to be aggregated on the server-side. In this strategy, the clients pack the weights to be
                aggregated along with a clipping bit calculated during training.
        Returns:
            Tuple[List[Tuple[NDArrays, int]], NDArrays]: The first tuple is the set of (weights, training counts) per
                client. The second is a set of clipping bits, one for each client.
        """
        weights_and_counts: List[Tuple[NDArrays, int]] = []
        clipping_bits: NDArrays = []
        for _, fit_res in results:
            sample_count = fit_res.num_examples
            updated_weights, clipping_bit = self.parameter_packer.unpack_parameters(
                parameters_to_ndarrays(fit_res.parameters)
            )
            weights_and_counts.append((updated_weights, sample_count))
            clipping_bits.append(np.array(clipping_bit))

        return weights_and_counts, clipping_bits

    def calculate_update_with_momentum(self, weights_update: NDArrays) -> None:
        """
        Performs a weight update with momentum. That is, combining some weighted value of the previous update with
        the current update.

        Args:
            weights_update (NDArrays): The current update after the weights have been aggregated from the training
            round.
        """
        if not self.m_t:
            self.m_t = weights_update
        else:
            self.m_t = [
                # NOTE: This is not normalized (beta vs. 1-beta) as used in the original implementation
                self.beta * prev_layer_update + noised_layer_update
                for prev_layer_update, noised_layer_update in zip(self.m_t, weights_update)
            ]

    def update_current_weights(self) -> None:
        """
        This function updates each of the layer weights using the server learning rate and the m_t values
        (computed with or without momentum).
        NOTE: It assumes that the values in m_t are UPDATES rather than raw weights.
        """
        assert self.m_t is not None
        self.current_weights = [
            current_layer_weight + self.server_learning_rate * layer_mt
            for current_layer_weight, layer_mt in zip(self.current_weights, self.m_t)
        ]

    def _update_clipping_bound_with_noised_bits(
        self,
        noised_clipping_bits: float,
    ) -> None:
        """
        Update the clipping bound help by the server given the noised aggregated clipping bits returned by the clients
        NOTE: The update formula may be found in the original paper.

        Args:
            noised_clipping_bits (float): This is the aggregated noised clipping bits derived from the clients.
        """
        self.clipping_bound = self.clipping_bound * math.exp(
            -self.clipping_learning_rate * (noised_clipping_bits - self.clipping_quantile)
        )

    def update_clipping_bound(self, clipping_bits: NDArrays) -> None:
        """
        This addes noise to the clipping bits returned by the clients and then updates the server-side clipping bound
        using this information.

        Args:
            clipping_bits (NDArrays): Bits associated with each of the clients. These are to be noised and aggregated
            in order to update the clipping bound on the server side.
        """
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
        they are theta_client - theta_server rather than just theta_client.
        NOTE: this function packs the clipping bound for clients as the last member of the parameters list.

        Args:
            server_round (int): Indicates the server round we're currently on.
            results (List[Tuple[ClientProxy, FitRes]]): The client identifiers and the results of their local training
                that need to be aggregated on the server-side. In this strategy, the clients pack the weights to be
                aggregated along with a clipping bit calculated during their local training cycle.
            failures (List[Union[Tuple[ClientProxy, FitRes], BaseException]]): These are the results and exceptions
                from clients that experienced an issue during training, such as timeouts or exceptions.

        Returns:
            Tuple[Optional[Parameters], Dict[str, Scalar]]: The aggregated model weights and the metrics dictionary.
                For this strategy, the server also packs a clipping bound to be sent to the clients. This is sent even
                if adaptive clipping is turned off and the value simply remains constant.
        """

        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # If first round compute total expected client weight
        if self.weighted_aggregation and server_round == 1:
            assert self.sample_counts is not None

            total_samples = sum(self.sample_counts)

            self.per_client_example_cap = (
                total_samples if self.per_client_example_cap is None else self.per_client_example_cap
            )

            self.total_client_weight: float = sum(
                [sample_count / self.per_client_example_cap for sample_count in self.sample_counts]
            )

        # Convert results with packed params of model weights and clipping bits
        weights_and_counts, clipping_bits = self.split_model_weights_and_clipping_bits(results)

        noise_multiplier = self.weight_noise_multiplier
        if self.adaptive_clipping:
            # The noise multiplier need only be modified in the event of using adaptive clipping to account for the
            # extra gradient information used to adapt the clipping threshold.
            noise_multiplier = self.modify_noise_multiplier()
            self.update_clipping_bound(clipping_bits)
            log(INFO, f"New Clipping Bound is: {self.clipping_bound}")

        if self.weighted_aggregation:
            assert self.per_client_example_cap is not None
            noised_aggregated_update = gaussian_noisy_weighted_aggregate(
                weights_and_counts,
                noise_multiplier,
                self.clipping_bound,
                self.fraction_fit,
                self.per_client_example_cap,
                self.total_client_weight,
            )
        else:
            noised_aggregated_update = gaussian_noisy_unweighted_aggregate(
                weights_and_counts,
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

        # Weights plus the clipping bound to be used by the clients
        packed_ndarrays = self.parameter_packer.pack_parameters(self.current_weights, self.clipping_bound)
        return ndarrays_to_parameters(packed_ndarrays), metrics_aggregated

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """
        This function configures a sample of clients for a training round. Due to the privacy accounting, this strategy
        requires that the sampling manager be of type BaseFractionSamplingManager.

        The function follows the standard configuration flow where the on_fit_config_fn function is used to produce
        configurations to be sent to all clients. These are packaged with the provided parameters and set over to the
        clients.

        Args:
            server_round (int): Indicates the server round we're currently on.
            parameters (Parameters): The parameters to be used to initialize the clients for the fit round.
            client_manager (ClientManager): The manager used to sample the clients. Currently we restrict this to
                be BaseFractionSamplingManager, which has a sample_fraction function built in.

        Returns:
            List[Tuple[ClientProxy, FitIns]]: List of sampled client identifiers and the configuration/parameters to
                be sent to each client (packaged as FitIns).
        """
        # This strategy requires the client manager to be of type at least BaseFractionSamplingManager
        assert isinstance(client_manager, BaseFractionSamplingManager)
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)

        fit_ins = FitIns(parameters, config)

        clients = client_manager.sample_fraction(self.fraction_fit, self.min_available_clients)

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """
        This function configures a sample of clients for an eval round. Due to the privacy accounting, this strategy
        requires that the sampling manager be of type BaseFractionSamplingManager.

        The function follows the standard configuration flow where the on_evaluate_config_fn function is used to
        produce configurations to be sent to all clients. These are packaged with the provided parameters and set over
        to the clients.

        Args:
            server_round (int): Indicates the server round we're currently on.
            parameters (Parameters): The parameters to be used to initialize the clients for the eval round.
            client_manager (ClientManager): The manager used to grab all of the clients. Currently we restrict this to
                be BaseFractionSamplingManager, which has a sample_fraction function built in.

        Returns:
            List[Tuple[ClientProxy, EvaluateIns]]: List of sampled client identifiers and the configuration/parameters
                to be sent to each client (packaged as EvaluateIns)
        """

        # This strategy requires the client manager to be of type at least BaseFractionSamplingManager
        assert isinstance(client_manager, BaseFractionSamplingManager)

        # Do not configure federated evaluation if fraction eval is 0 or server is not initialized
        if self.fraction_evaluate == 0.0:
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
