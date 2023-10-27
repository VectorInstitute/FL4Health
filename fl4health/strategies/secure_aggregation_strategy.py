from fl4health.strategies.basic_fedavg import BasicFedAvg
from flwr.server.client_manager import ClientManager
from flwr.common import Parameters, FitIns, Scalar, MetricsAggregationFn, NDArrays
from typing import List, Tuple
from flwr.server.client_proxy import ClientProxy
from flwr.common import GetPropertiesIns
from typing import Optional, Callable, Dict
from fl4health.client_managers.base_sampling_manager import BaseFractionSamplingManager

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

    
    def package_request(self, request: Dict[str, Scalar], event_name: str, client_manager: ClientManager) -> Request:

        assert client_manager.num_available() > 1   # make sure there are clients online

        # get all online clients for SecAgg (substitute for different client sampler for SecAgg+)
        if isinstance(client_manager, BaseFractionSamplingManager):
            clients_proxy_list = client_manager.sample_all(min_num_clients=self.min_available_clients)
        else:
            # Grab all available clients using the basic Flower client manager
            num_available_clients = client_manager.num_available()
            clients_proxy_list = client_manager.sample(num_available_clients, min_num_clients=self.min_available_clients)

        # adjoin event_name
        req = {
             "event_name": event_name,
             **request
        }

        # Package dictionary to Flower request format 
        wrapper = GetPropertiesIns(req)

        packaged = map(lambda client: (client, wrapper), clients_proxy_list)
        return list(packaged)

