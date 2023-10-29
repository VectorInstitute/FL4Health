from fl4health.server.base_server import FlServerWithCheckpointing, ExchangerType
from flwr.server.client_manager import ClientManager
from fl4health.strategies.secure_aggregation_strategy import SecureAggregationStrategy
from typing import Optional, Dict, Any
from fl4health.reporting.fl_wanb import ServerWandBReporter
from fl4health.checkpointing.checkpointer import TorchCheckpointer
from flwr.server.history import History
from torch.nn import Module
from logging import DEBUG, INFO, WARNING
from flwr.common.logger import log
from fl4health.strategies.strategy_with_poll import StrategyWithPolling
from .polling import poll_clients
import timeit
from flwr.common.typing import Scalar
from fl4health.security.secure_aggregation import ServerCryptoKit, Event
from time import sleep
from fl4health.client_managers.base_sampling_manager import BaseFractionSamplingManager



class SecureAggregationServer(FlServerWithCheckpointing):

    def __init__(
        self, 
        client_manager: ClientManager, 
        strategy: SecureAggregationStrategy, 
        model: Module,
        parameter_exchanger: ExchangerType,
        wandb_reporter: Optional[ServerWandBReporter] = None,
        checkpointer: Optional[TorchCheckpointer] = None,
        # secure aggregation params
        shamir_reconstruction_threshold: int = 2,
        arithmetic_modulus: int = 1 << 30,
    ) -> None:

        assert isinstance(strategy, SecureAggregationStrategy)    

        super().__init__(client_manager, model, parameter_exchanger, wandb_reporter, strategy, checkpointer)

        # federated round
        self.fl_round = 0 

        # Handled in the respective communication stack, as in polling.py
        self.timeout: Optional[float] = None 

        # handles SecAgg cryptography on the server-side
        self.crypto = ServerCryptoKit()
        assert isinstance(shamir_reconstruction_threshold, int) and shamir_reconstruction_threshold >= 2
        self.crypto.set_shamir_threshold(shamir_reconstruction_threshold)

        assert isinstance(arithmetic_modulus, int) and arithmetic_modulus > 1
        self.crypto.set_arithmetic_modulus(arithmetic_modulus)



    def set_timeout(self, timeout: float) -> None:
        self.timeout = timeout

    def fit(self, num_rounds: int, timeout: Optional[float]) -> History:
        # ============= event sequence ===============
        """Run federated averaging for a number of rounds."""
        history = History()

        # Initialize parameters
        log(INFO, "Initializing global parameters")
        self.parameters = self._get_initial_parameters(timeout=timeout)
        log(INFO, "Evaluating initial parameters")
        res = self.strategy.evaluate(0, parameters=self.parameters)
        if res is not None:
            log(
                INFO,
                "initial parameters (loss, other metrics): %s, %s",
                res[0],
                res[1],
            )
            history.add_loss_centralized(server_round=0, loss=res[0])
            history.add_metrics_centralized(server_round=0, metrics=res[1])

        # Run federated learning for num_rounds
        log(INFO, "FL starting")
        start_time = timeit.default_timer()

        for current_round in range(1, num_rounds + 1):
            if current_round > 1:
                # call client
                self.api(request_dict={"greet": "server greets client", "round": current_round}, event_name="greet")
                self.secure_aggregation()

            # Train model and replace previous global model
            res_fit = self.fit_round(server_round=current_round, timeout=timeout)
            if res_fit:
                parameters_prime, fit_metrics, _ = res_fit  # fit_metrics_aggregated
                if parameters_prime:
                    self.parameters = parameters_prime
                history.add_metrics_distributed_fit(
                    server_round=current_round, metrics=fit_metrics
                )

            # Evaluate model using strategy implementation
            res_cen = self.strategy.evaluate(current_round, parameters=self.parameters)
            if res_cen is not None:
                loss_cen, metrics_cen = res_cen
                log(
                    INFO,
                    "fit progress: (%s, %s, %s, %s)",
                    current_round,
                    loss_cen,
                    metrics_cen,
                    timeit.default_timer() - start_time,
                )
                history.add_loss_centralized(server_round=current_round, loss=loss_cen)
                history.add_metrics_centralized(
                    server_round=current_round, metrics=metrics_cen
                )

            # Evaluate model on a sample of available clients
            res_fed = self.evaluate_round(server_round=current_round, timeout=timeout)
            if res_fed:
                loss_fed, evaluate_metrics_fed, _ = res_fed
                if loss_fed:
                    history.add_loss_distributed(
                        server_round=current_round, loss=loss_fed
                    )
                    history.add_metrics_distributed(
                        server_round=current_round, metrics=evaluate_metrics_fed
                    )

        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        log(INFO, "FL finished in %s", elapsed)
    

        if self.wandb_reporter:
            self.wandb_reporter.report_metrics(num_rounds, history)
        return history
    
    def secure_aggregation(self):
        
        N = self.get_peer_number()
        # assert 1 <= N <= self.crypto.arithmetic_modulus
        # assert 2 <= self.crypto.shamir_reconstruction_threshold <= N    
        self.crypto.set_number_of_bobs(N)
        self.broadcast_id()
        # c = self._client_manager.all()
        # keys = list(c.keys())
        # self.debugger(keys)
        # sleep(3)
        # self.debugger(c[keys[0]])
        # self.debugger(type(c[keys[0]]))



        # time.sleep(10)
        # round 0, advertise keys
        req_keys = {
            # meta data 
            'sender': 'server',
            'fl_round': self.fl_round,
        }

        """
        Expected dict of structure
        {
            
        }
        """
        public_keys = self.api(request_dict=req_keys, event_name=Event.ADVERTISE_KEYS.value)
        
    def debugger(self, info):
        log(DEBUG, 6*'\n')
        log(DEBUG, info)


    def get_peer_number(self):
        """The number of peers of Alice (total number of clients - 1)"""
        return self._client_manager.num_available() - 1

    def api(self, request_dict: Dict[str, Scalar], event_name: str) -> Any:
        """
        How this works:

        1. Call this method on the server side to pass request_dict to all the client.
        Secure aggregation uses events

        class Event(Enum):
            ADVERTISE_KEYS = 'round 0'
            SHARE_KEYS = 'round 1'
            MASKED_INPUT_COLLECTION = 'round 2'
            UNMASKING = 'round 4'
            BROADCAST_ID = 'assign client_integer'


        The event_value: str defines how client handles the API call. Thus we pass Event.UNMASING.value
        to this api() method as opposed to the enum Event.UNMASING.

        2. Clients receive the request_dict in SecureAggregationClient.get_properties().

        3. This SecureAggregationServer.api() method returns online clients' response.
        """
        log(INFO, "API calling client \n")

        request = self.strategy.package_request(request_dict, event_name, self._client_manager)

        # later we will extend the SecAgg protocol implementation to deal with dropouts
        online, dropouts = poll_clients(
            client_instructions=request,
            max_workers=self.max_workers,
            timeout=self.timeout
        )
        
        # resolve response
        response = []
        i = 1
        # NOTE for test purposes we will discard the client information, and only take their response
        for client, res in online:
            # type of online variable is List[Tuple[ClientProxy, GetPropertiesRes]]
            response.append(res.properties)
            i += 1
        return response

    def broadcast_id(self) -> int:
        assert isinstance(self.strategy, SecureAggregationStrategy)

        if isinstance(self._client_manager, BaseFractionSamplingManager):
            clients_list = self._client_manager.sample_all(min_num_clients=self.strategy.min_available_clients)
        else:
            # Grab all available clients using the basic Flower client manager
            num_available_clients = self._client_manager.num_available()
            clients_list = self._client_manager.sample(num_available_clients, min_num_clients=self.strategy.min_available_clients)
        
        all_requests = []
        client_int = 1
        for client in clients_list:
            assert client.cid is not None
            self.crypto.append_client_table(client_ip=client.cid, client_id=client_int)
            req_dict = {
                'sender': 'server',
                'fl_round': self.fl_round,
                'client_integer': client_int,
            }
            request = self.strategy.package_single_client_request(
                client=client,
                request=req_dict,
                event_name=Event.BROADCAST_ID.value
            )
            all_requests.append(request)
            client_int += 1

        # broadcast
        online, dropouts = poll_clients(
            client_instructions=all_requests,
            max_workers=self.max_workers,
            timeout=self.timeout
        )
        return len(online)
        