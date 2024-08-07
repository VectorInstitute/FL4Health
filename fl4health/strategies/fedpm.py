from collections import defaultdict
from functools import reduce
from typing import Callable, DefaultDict, Dict, List, Optional, Tuple

import numpy as np
from flwr.common import MetricsAggregationFn, NDArray, NDArrays, Parameters
from flwr.common.typing import Scalar

from fl4health.strategies.fedavg_dynamic_layer import FedAvgDynamicLayer


class FedPm(FedAvgDynamicLayer):
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
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
        weighted_eval_losses: bool = True,
        bayesian_aggregation: bool = True,
    ) -> None:
        """
        A strategy that is used for aggregating probability masks in the "Federated Probabilistic Mask Training"
        paradigm, as detailed in http://arxiv.org/pdf/2209.15328. The implementation here allows for simply averaging
        the probability masks, as well as the more sophisticated Bayesian aggregation approach.

        Note: since the parameters aggregated by this strategy are supposed to be binary masks, by default
        FedPM performs uniformed averaging. The effect of weighted averaging is also not covered in the original work.

        Args:
            fraction_fit (float, optional): Fraction of clients used during training. Defaults to 1.0. Defaults to 1.0.
            fraction_evaluate (float, optional): Fraction of clients used during validation. Defaults to 1.0.
            min_fit_clients (int, optional): Minimum number of clients used during fitting. Defaults to 2.
            min_evaluate_clients (int, optional): Minimum number of clients used during validation. Defaults to 2.
            min_available_clients (int, optional): Minimum number of clients used during validation. Defaults to 2.
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
            initial_parameters (Optional[Parameters], optional): Initial global model parameters. Defaults to None.
            fit_metrics_aggregation_fn (Optional[MetricsAggregationFn], optional): Metrics aggregation function.
                Defaults to None.
            evaluate_metrics_aggregation_fn (Optional[MetricsAggregationFn], optional): Metrics aggregation function.
                Defaults to None.
            weighted_aggregation (bool, optional): Determines whether parameter aggregation is a linearly weighted
                average or a uniform average. FedAvg default is weighted average by client dataset counts.
                Defaults to True.
            weighted_eval_losses (bool, optional): Determines whether losses during evaluation are linearly weighted
                averages or a uniform average. FedAvg default is weighted average of the losses by client dataset
                counts. Defaults to True.
            bayesian_aggregation (bool): Determines whether Bayesian aggregation is used.
        """
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
            weighted_aggregation=False,
            weighted_eval_losses=weighted_eval_losses,
        )
        # Parameters for Beta distribution.
        self.beta_parameters: Dict[str, Tuple[NDArray, NDArray]] = {}
        self.bayesian_aggregation = bayesian_aggregation

    def aggregate(self, results: List[Tuple[NDArrays, int]]) -> Dict[str, NDArray]:
        if not self.bayesian_aggregation:
            return super().aggregate(results)
        else:
            return self.aggregate_bayesian(results)

    def aggregate_bayesian(self, results: List[Tuple[NDArrays, int]]) -> Dict[str, NDArray]:
        """
        Perform posterior update to the Beta distribution parameters based on the binary masks
        sent by the clients.

        More precisely, each client maintains for each one of its parameter tensors
        a "probability score tensor". These scores (after applying the Sigmoid function to them)
        are Bernoulli probabilities which indicate how likely their corresponding parameters are
        to be pruned or kept. Each client samples a binary mask for every one of its parameter tensors
        based on the corresponding Bernoulli probabilities. These masks are sent to the server
        for aggregation.

        Here, we assume that the bernoulli probabilities of each client themselves follow a Beta
        distribution with parameters alpha and beta. Then the binary masks may be viewed as data that
        can be used to update alpha and beta, and this corresponds to a posterior update.
        Due to the conjugate relation between the Beta and Bernoulli distributions, the posterior distribution
        is still a Beta distribution, so we can perform the aggregation in this manner every round.

        In this case, the updates performed are:
            alpha_new = alpha + M
            beta_new = beta + K * 1 - M
            theta = (alpha_new - 1) / (alpha_new + beta_new - 2)
        where M is the sum of all binary masks corresponding to a particular parameter tensor,
        K is the number of clients, and "1" in the second equation refers to an array of all
        ones of the same shape as M.

        In the beginning, alpha and beta are initialized to arrays of all ones.
        """
        names_to_layers: DefaultDict[str, List[NDArray]] = defaultdict(list)
        total_num_clients: DefaultDict[str, int] = defaultdict(int)

        # unpack the parameters and initialize the beta parameters to be all ones if they have not already
        # been initialized.
        for packed_layers, _ in results:
            layers, names = self.parameter_packer.unpack_parameters(packed_layers)
            for layer, name in zip(layers, names):
                names_to_layers[name].append(layer)
                total_num_clients[name] += 1
                if name not in self.beta_parameters:
                    alpha = np.ones_like(layer)
                    beta = np.ones_like(layer)
                    self.beta_parameters[name] = (alpha, beta)

        aggregation_result: Dict[str, NDArray] = {}

        # posterior update of the beta parameters and using them
        # to compute the final result.
        for parameter_name in self.beta_parameters.keys():
            m_agg = reduce(np.add, names_to_layers[parameter_name])
            n_clients = total_num_clients[parameter_name]
            alpha, beta = self.beta_parameters[parameter_name]
            alpha_new = alpha + m_agg
            beta_new = beta + np.ones_like(beta) * n_clients - m_agg
            self.beta_parameters[parameter_name] = (alpha_new, beta_new)
            aggregation_result[parameter_name] = (alpha_new - 1) / (alpha_new + beta_new - 2)

        return aggregation_result

    def reset_beta_priors(self) -> None:
        """
        Reset the alpha and beta parameters for the Beta distribution to be arrays of all ones.
        """
        for parameter_name in self.beta_parameters.keys():
            alpha, beta = self.beta_parameters[parameter_name]
            self.beta_parameters[parameter_name] = (np.ones_like(alpha), np.ones_like(beta))
