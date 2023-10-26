from fl4health.strategies.basic_fedavg import BasicFedAvg
from flwr.server.client_manager import ClientManager
from flwr.common import Parameters, FitIns, Scalar, MetricsAggregationFn, NDArrays
from typing import List, Tuple
from flwr.server.client_proxy import ClientProxy
from flwr.common import GetPropertiesIns
from typing import Optional, Callable, Dict

Request = List[Tuple[ClientProxy, GetPropertiesIns]]

class SecureAggregationStrategy(BasicFedAvg):

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
        weighted_aggregation: bool = True,
        weighted_eval_losses: bool = True,
    ) -> None:
            # NOTE currently secure aggregation supports no dropouts, hence fraction_fit_fit = 1
            # I will define on_fit_config_fn to avoid complication
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
                weighted_eval_losses= weighted_eval_losses,
            )

    
    def package_request(self, request: Dict[str, Scalar], client_manager: ClientManager) -> Request:

        assert client_manager.num_available() > 1   # make sure there are clients online

        # all online clients for SecAgg (substitute for different sampler for SecAgg+)
        clients_proxy_list = client_manager.all()

        # Package dictionary to Flower request format 
        wrapper = GetPropertiesIns(request)

        packaged = map(lambda client: (client, wrapper), clients_proxy_list)
        return list(packaged)

