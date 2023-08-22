from typing import Callable, Dict, List, Optional, Tuple

from flwr.common import MetricsAggregationFn
from flwr.common.typing import GetPropertiesIns, NDArrays, Parameters, Scalar
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from fl4health.client_managers.base_sampling_manager import BaseSamplingManager
from fl4health.strategies.fedavg_sampling import FedAvgSampling


class InstanceLevelDPFedAvgSampling(FedAvgSampling):
    """
    Instance Level Differentially Private Federated Averaging Strategy
    """

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
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None
    ) -> None:
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

    def configure_poll(
        self, server_round: int, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, GetPropertiesIns]]:
        """Configure server for polling of clients."""

        # This strategy requires the client manager to be of type at least BaseSamplingManager
        assert isinstance(client_manager, BaseSamplingManager)
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)

        property_ins = GetPropertiesIns(config)

        clients = client_manager.sample_all(min_num_clients=self.min_available_clients)

        # Return client/config pairs
        return [(client, property_ins) for client in clients]
