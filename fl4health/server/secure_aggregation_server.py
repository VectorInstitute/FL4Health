import pickle
import timeit
from logging import DEBUG, INFO
from typing import Any, Dict, List, Optional

from flwr.common.logger import log
from flwr.common.typing import Scalar
from flwr.server.client_manager import ClientManager, ClientProxy
from flwr.server.history import History
from torch.nn import Module

from fl4health.checkpointing.checkpointer import TorchCheckpointer
from fl4health.client_managers.base_sampling_manager import BaseFractionSamplingManager
from fl4health.reporting.fl_wanb import ServerWandBReporter
from fl4health.security.secure_aggregation import ClientId, Event, ServerCryptoKit
from fl4health.server.base_server import ExchangerType, FlServerWithCheckpointing
from fl4health.server.polling import poll_clients
from fl4health.strategies.secure_aggregation_strategy import SecureAggregationStrategy


class SecureAggregationServer(FlServerWithCheckpointing):
    """
    Server supporting secure aggregation.
    """

    def __init__(
        self,
        client_manager: ClientManager,
        strategy: SecureAggregationStrategy,
        model: Module,
        parameter_exchanger: ExchangerType,
        wandb_reporter: Optional[ServerWandBReporter] = None,
        checkpointer: Optional[TorchCheckpointer] = None,
        timeout: Optional[float] = 30,
        # secure aggregation params
        shamir_reconstruction_threshold: int = 2,
        arithmetic_modulus: int = 1 << 30,
    ) -> None:

        # Ensure cryptography runs correctly.
        assert isinstance(strategy, SecureAggregationStrategy)
        assert isinstance(arithmetic_modulus, int) and arithmetic_modulus > 1
        assert isinstance(shamir_reconstruction_threshold, int) and shamir_reconstruction_threshold >= 2

        super().__init__(client_manager, model, parameter_exchanger, wandb_reporter, strategy, checkpointer)

        self.crypto = ServerCryptoKit()

        # We may also use these setters to adjust ServerCryptoKit params during FL as needed,
        # depending on dropout severity.
        self.crypto.set_shamir_threshold(shamir_reconstruction_threshold)
        self.crypto.set_arithmetic_modulus(arithmetic_modulus)

        self.current_fl_round = 0
        self.timeout = timeout

    def api(self, request_dict: Dict[str, Scalar], event_name: str) -> Any:
        """This server-side API streamlines communication with the client during SecAgg.

        1. Call this method on the server side to pass request_dict to all clients.
        Secure aggregation uses events

        class Event(Enum):
            ADVERTISE_KEYS = 'round 0'
            SHARE_KEYS = 'round 1'
            MASKED_INPUT_COLLECTION = 'round 2'
            UNMASKING = 'round 4'

        The event_value: str defines how client handles the API call. Thus we pass Event.UNMASING.value
        to this api() method as opposed to the enum Event.UNMASING.

        2. Clients receive the request_dict in SecureAggregationClient.get_properties().

        3. This SecureAggregationServer.api() method returns online clients' response.
        """

        log(INFO, "API calling all clients...")

        request = self.strategy.package_request(request_dict, event_name, self._client_manager)

        # TODO SecAgg protocol implementation will be extended to deal with dropouts in the future.
        online, dropouts = poll_clients(
            client_instructions=request, max_workers=self.max_workers, timeout=self.timeout
        )

        # resolve responses.
        responses = [res.properties for client_proxy, res in online]
        # If needed, perform custom processing or use client identity via client_proxy before return.

        return responses

    def fit(self, num_rounds: int, timeout: Optional[float]) -> History:
        """
        Server orchestrated federated learning rounds.
        """

        # We keep record of distributed (and not centralized) loss / metrics in this application.
        history = History()

        log(INFO, "Initializing global parameters...")
        self.parameters = self.strategy.initial_parameters
        log(INFO, "...done, global parameters initialized.")

        log(INFO, "Starting FL...")
        start_time = timeit.default_timer()

        # ------------------------------------------------------------------------------------------
        log(INFO, f"Clients available before: {self.client_manager().num_available()}")
        # self.client_manager().wait_for(3, timeout=timeout)
        self.fit_round(server_round=0, timeout=timeout)
        log(INFO, f"Clients available after: {self.client_manager().num_available()}")
        """
        Server round 0 is designated to survey available clients; without at least one fit_round()
        call, the client manager sees 0 available clients, and key exchange cannot be processed.
        This round is specially processed by the client without training, see client fit() method.
        """
        # ------------------------------------------------------------------------------------------

        # Complete federated learning rounds under the Secure Aggregation Protocol.
        for current_round in range(1, num_rounds + 1):
            self.current_fl_round = current_round

            # verification phase
            N = self.get_peer_number()
            # NOTE In dropout case, we can optionally reset the threshold depending on N.
            assert 2 <= self.crypto.shamir_reconstruction_threshold <= N
            self.crypto.set_number_of_bobs(N)  # be sure to set bobs/peer number before key agreement

            self.advertise_keys()
            self.share_keys()
            self.masked_input_collection()
            res_fit = self.unmasking(current_round=current_round, timeout=timeout)

            if res_fit:
                parameters_prime, fit_metrics, _ = res_fit
                if parameters_prime:
                    # TODO If necessary, handle dropouts before updating global parameters.
                    self.parameters = parameters_prime
                history.add_metrics_distributed_fit(server_round=current_round, metrics=fit_metrics)

            # Evaluate model on a sample of available clients
            res_fed = self.evaluate_round(server_round=current_round, timeout=timeout)
            if res_fed:
                loss_fed, evaluate_metrics_fed, _ = res_fed
                if loss_fed:
                    history.add_loss_distributed(server_round=current_round, loss=loss_fed)
                    history.add_metrics_distributed(server_round=current_round, metrics=evaluate_metrics_fed)

        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        log(INFO, "...FL finished in %s", elapsed)

        if self.wandb_reporter:
            self.wandb_reporter.report_metrics(num_rounds, history)
        return history

    """
    The following auxiliary methods for Secure Aggregation are defined in the order they are called by the fit() method.
    """

    def get_peer_number(self):
        """The number of peers of Alice (total number of clients - 1)"""
        return self._client_manager.num_available() - 1

    def get_metadata(self) -> Dict[str, Scalar]:
        assert self.crypto.number_of_bobs is not None
        return {
            "sender": "server",
            "current_fl_round": self.current_fl_round,
            "arithmetic_modulus": self.crypto.arithmetic_modulus,
            "shamir_reconstruction_threshold": self.crypto.shamir_reconstruction_threshold,
            "number_of_bobs": self.crypto.number_of_bobs,
        }

    def get_client_list(self) -> List[ClientProxy]:
        """Get all available clients"""
        if isinstance(self._client_manager, BaseFractionSamplingManager):
            clients_list = self._client_manager.sample_all(min_num_clients=self.strategy.min_available_clients)
        else:
            # Grab all available clients using the basic Flower client manager
            num_available_clients = self._client_manager.num_available()
            clients_list = self._client_manager.sample(
                num_available_clients, min_num_clients=self.strategy.min_available_clients
            )
        return clients_list

    def advertise_keys(self) -> None:
        """
        Round 0 of the Secure Aggregation Protocol.
        """
        # assign client integer, one by one
        all_requests = []
        client_int = 1
        for client in self.get_client_list():
            assert client.cid
            self.crypto.append_client_table(client_ip=client.cid, client_id=client_int)
            req_dict = {
                **self.get_metadata(),
                "client_integer": client_int,
            }
            request = self.strategy.package_single_client_request(
                client=client, request=req_dict, event_name=Event.ADVERTISE_KEYS.value
            )
            all_requests.append(request)
            client_int += 1

        # broadcast
        online, dropouts = poll_clients(
            client_instructions=all_requests, max_workers=self.max_workers, timeout=self.timeout
        )

        assert len(online) >= self.crypto.shamir_reconstruction_threshold

        # parse client responses to record public key shares
        for client, response in online:
            ip, res = client.cid, response.properties

            assert res["sender"] == "client"
            assert res["event_name"] == Event.ADVERTISE_KEYS.value

            self.crypto.append_client_public_keys(
                client_integer=res["client_integer"],
                encryption_public_key=res["public_encryption_key"],
                masking_public_key=res["public_mask_key"],
            )

    def share_keys(self) -> None:
        """
        Round 1 of the Secure Aggregation Protocol.
        """
        pickled_byte_keys = pickle.dumps(self.crypto.get_all_public_keys())
        req = {**self.get_metadata(), "bobs_public_keys": pickled_byte_keys}
        res = self.api(request_dict=req, event_name=Event.SHARE_KEYS.value)
        # TODO handle shamir shares

        # shamir_secrets = res['serialized_encrypted_shamir']

        # Expected type
        # {
        #     ClientId: {
        #         'encrypted_shamir_pairwise': EncryptedShamirSecret,
        #         'encrypted_shamir_self': EncryptedShamirSecret
        #     },
        # }

    def masked_input_collection(self):
        pass

    def unmasking(self, current_round, timeout):
        # Aggregated parameter update

        # remove self/pair mask
        res_fit = self.fit_round(server_round=current_round, timeout=timeout)

    def pre_secure_aggregation(self):
        N = self.get_peer_number()
        self.debugger(f"secure agg has received {N+1} clients")

        # assert 1 <= N <= self.crypto.arithmetic_modulus
        # assert 2 <= self.crypto.shamir_reconstruction_threshold <= N
        self.crypto.set_number_of_bobs(N)

        # list of dictionaries, each dict contains keys ['client_integer', 'encryption_key', 'mask_key']
        all_public_keys = self.setup_and_key_agreement()
        shamir_shares = self.broadcast_keys()
        self.crypto.clear_cache()

    def debugger(self, *info):
        log(DEBUG, 6 * "\n")
        for item in info:
            log(DEBUG, item)

    def request_masked_input(self):
        req = {"sender": "server", "fl_round": self.fl_round, "even_name": Event.MASKED_INPUT_COLLECTION.value}
