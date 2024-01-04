import pickle
import timeit
from dataclasses import dataclass
from logging import DEBUG, INFO
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from flwr.common import GetPropertiesIns
from flwr.common.logger import log
from flwr.common.typing import Scalar
from flwr.server.client_manager import ClientManager, ClientProxy
from flwr.server.history import History
from torch.nn import Module

from fl4health.checkpointing.checkpointer import TorchCheckpointer
from fl4health.client_managers.base_sampling_manager import BaseFractionSamplingManager
from fl4health.reporting.fl_wanb import ServerWandBReporter
from fl4health.security.secure_aggregation import ClientId, DestinationClientId, Event, ServerCryptoKit, ShamirOwnerId
from fl4health.server.base_server import ExchangerType, FlServerWithCheckpointing
from fl4health.server.polling import poll_clients
from fl4health.strategies.secure_aggregation_strategy import SecureAggregationStrategy


@dataclass
class Status:

    dropout: bool = True


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

        # Maps client integer to client proxy. Used by self.api() in the track_client = True mode.
        self.id_proxy_table: Dict[ClientId, ClientProxy] = {}

        self.client_status: Dict[ClientId, Status] = {}

    def set_status(self) -> None:
        N = self.client_manager().num_available()
        assert N > 0
        for id in range(1, 1 + N):
            client = Status(
                dropout=True
            )  # when api (with track_client option is on) receives message, we set dropout = False
            self.client_status[id] = client

    def set_online(self, client_integer: ClientId) -> None:
        assert client_integer in self.client_status
        self.client_status[client_integer].dropout = False

    def api(self, request_dict: Dict[str, Scalar], event_name: str, track_clients=False) -> List[Dict[str, Scalar]]:
        self.debugger(event_name)
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

        isinstance(self.strategy, SecureAggregationStrategy)
        log(INFO, "API calling all clients...")

        request = self.strategy.package_request(request_dict, event_name, self._client_manager)

        # TODO SecAgg protocol implementation will be extended to deal with dropouts in the future.
        online, dropouts = poll_clients(
            client_instructions=request, max_workers=self.max_workers, timeout=self.timeout
        )

        responses = []

        # Resolve responses.
        if track_clients:

            # clear table from previous rounds
            self.id_proxy_table = {}

            for client_proxy, res in online:
                client_response = res.properties
                assert "client_integer" in client_response
                assert client_response["client_integer"] is not None
                client_integer = client_response["client_integer"]
                self.id_proxy_table[client_integer] = client_proxy
                responses.append(client_response)
                self.set_online(client_integer)
        else:
            # does not store client proxy
            for client_proxy, res in online:
                data = res.properties
                responses.append(data)
                self.set_online(data["client_integer"])

        return responses

    def api_custom_message(self, custom_messages: Dict[ClientId, Dict], event_name: str, track_clients=False) -> None:
        """This API allows the server to send messages customized for each client.
        Be careful not to deeply nest the inner dictionary as it can cause slient errors. If need be, pickle the innter dictionary.
        """

        requests: List[Tuple[ClientProxy, GetPropertiesIns]] = []

        for client_integer, client_dict in custom_messages.items():
            self.debugger(client_dict, type(client_dict))
            client_proxy = self.id_proxy_table[client_integer]
            client_req = GetPropertiesIns({"event_name": event_name, "pickled_message": pickle.dumps(client_dict)})
            requests.append((client_proxy, client_req))

        online, dropouts = poll_clients(
            client_instructions=requests, max_workers=self.max_workers, timeout=self.timeout
        )

        responses = []

        # Resolve responses.
        if track_clients:

            # clear table from previous rounds
            self.id_proxy_table = {}

            for client_proxy, res in online:
                client_response = res.properties
                assert "client_integer" in client_response
                assert client_response["client_integer"] is not None
                client_integer = client_response["client_integer"]
                self.id_proxy_table[client_integer] = client_proxy
                responses.append(client_response)
                self.set_online(client_integer)
        else:
            # does not store client proxy
            for client_proxy, res in online:
                data = res.properties
                responses.append(data)
                self.set_online(data["client_integer"])

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

            self.set_status()

            self.advertise_keys()
            self.share_keys()
            self.masked_input_collection(current_round=current_round, timeout=timeout)
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
        Server communicates with clients 2x during this round:
            1. ask clients to send their Shamir shares
            2. ask clients to receive others' Shamir shares
        """
        pickled_byte_keys = pickle.dumps(self.crypto.get_all_public_keys())
        req = {**self.get_metadata(), "bobs_public_keys": pickled_byte_keys}

        # array of responses
        res: List[Dict[str, Scalar]] = self.api(
            request_dict=req, event_name=Event.SHARE_KEYS.value, track_clients=True
        )

        # make this initialization depend on online clients
        N = self.client_manager().num_available()

        assert N >= self.crypto.shamir_reconstruction_threshold

        shamir_delivery = {i: {} for i in range(1, 1 + N)}  # note that client integers > 0
        """The structure of shamir_delivery is
        {
            DestinationClientId: {
                ShamirOwnerId: {
                    'encrypted_shamir_pairwise': bytes,
                    'encrypted_shamir_self': bytes
                }
            }
        }
        """

        for res_dict in res:
            owner_id = res_dict["client_integer"]
            shamir_shares: Dict = pickle.loads(res_dict["serialized_encrypted_shamir"])
            """
            Strucure of shamir_shares
            {
                ClientId: {
                    'encrypted_shamir_pairwise': bytes,
                    'encrypted_shamir_self': bytes
                },
            }
            """
            for destination_id, encrypted_shamir_shares in shamir_shares.items():
                shamir_delivery[destination_id][owner_id] = {
                    "encrypted_shamir_pairwise": encrypted_shamir_shares["encrypted_shamir_pairwise"],
                    "encrypted_shamir_self": encrypted_shamir_shares["encrypted_shamir_self"],
                }

        res = self.api_custom_message(custom_messages=shamir_delivery, event_name=Event.MASKED_INPUT_COLLECTION.value)
        # self.debugger('here is the delivery', shamir_delivery)
        # fix !!! revise poll_clients to talk to specific clients
        # res = self.api(request_dict={'items': pickle.dumps(shamir_delivery)}, event_name=Event.MASKED_INPUT_COLLECTION.value)
        # self.api(request_dict={'items': {'hi':1}}, event_name=Event.MASKED_INPUT_COLLECTION.value)

    def unmasking(self, model_vect: np.ndarray):
        """Outputs sum of masks to be removed (subtracted)."""

        # if bob has dropped out, the pair mask Shamir secret will be shared by alice
        # if bob is online, the self mask Shamir secret will be shared by alice

        requests = []
        for id, status in self.client_status.items():
            req = (id, status.dropout)
            requests.append(req)

        res = self.api(request_dict={"dropout_status": pickle.dumps(requests)}, event_name=Event.UNMASKING.value)

        shamir_secrets = {id: [] for id in self.client_status.keys()}

        for res_dict in res:
            secrets = pickle.loads(res_dict["shamir_secrets"])
            for id, secret in secrets:
                shamir_secrets[id].append(secret)

        seeds = map(
            lambda shares: self.crypto.shamir_reconstruct_secret(shares, self.crypto.shamir_reconstruction_threshold),
            [shares for shares in shamir_secrets.values()],
        )

        dim = model_vect.size()

        masks = np.zeros(dim)
        for seed in seeds:
            masks += self.crypto.reconstruct_mask(seed, dim)

        return model_vect - masks

        # server performs shamir reconstruction of the seeds and generation and summation of masks

    def masked_input_collection(self, current_round, timeout):
        # Aggregated parameter update
        return self.fit_round(server_round=current_round, timeout=timeout)

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
