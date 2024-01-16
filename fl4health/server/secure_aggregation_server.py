import pickle
import timeit
from dataclasses import dataclass
from itertools import product
from logging import DEBUG, INFO, WARN
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from flwr.common import GetPropertiesIns, Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.logger import log
from flwr.common.typing import NDArrays, Scalar
from flwr.server.client_manager import ClientManager, ClientProxy
from flwr.server.history import History
from flwr.server.server import FitResultsAndFailures, fit_clients
from torch.nn import Module

from fl4health.checkpointing.checkpointer import TorchCheckpointer
from fl4health.client_managers.base_sampling_manager import BaseFractionSamplingManager
from fl4health.privacy_mechanisms.discrete_gaussian_mechanism import (
    generate_sign_diagonal_matrix,
    generate_walsh_hadamard_matrix,
)
from fl4health.privacy_mechanisms.index import PrivacyMechanismIndex
from fl4health.reporting.fl_wanb import ServerWandBReporter
from fl4health.reporting.secure_aggregation_blackbox import BlackBox
from fl4health.security.secure_aggregation import (
    ClientId,
    DestinationClientId,
    EllipticCurvePrivateKey,
    Event,
    Seed,
    ServerCryptoKit,
    ShamirOwnerId,
)
from fl4health.server.base_server import ExchangerType, FlServerWithCheckpointing
from fl4health.server.polling import poll_clients
from fl4health.server.secure_aggregation_utils import get_model_dimension, unvectorize_model, vectorize_model
from fl4health.strategies.secure_aggregation_strategy import SecureAggregationStrategy


@dataclass
class Status:
    dropout: bool


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
        model_integer_range: int = 1 << 30,
        dropout_mode=False,
    ) -> None:

        assert isinstance(strategy, SecureAggregationStrategy)
        super().__init__(client_manager, model, parameter_exchanger, wandb_reporter, strategy, checkpointer)
        self.timeout = timeout
        self.dropout_mode = dropout_mode

        self.crypto = ServerCryptoKit()
        self.set_shamir_threshold(shamir_reconstruction_threshold)
        self.set_model_integer_range(model_integer_range)
        self.crypto.model_dimension = get_model_dimension(model)

        self.blackbox = BlackBox()

        # for communication with specific client
        self.id_proxy_table: Dict[ClientId, ClientProxy] = {}
        self.cid_to_id_table: Dict[str, ClientId] = {}

        # did client dropout
        self.id_status_table: Dict[ClientId, Status] = {}

        # differential privacy
        self.privacy_settings = {
            "dp_mechanism": PrivacyMechanismIndex.DiscreteGaussian.value,
            "noise_scale": 1,
            "granularity": 1,
            "clipping_threshold": 10,
            "bias": 0.5,
        }

    # the 'main' function; orchestrates FL
    def fit(self, num_rounds: int, timeout: Optional[float]) -> History:

        log(INFO, "Initializing global parameters & starting FL.")

        self.parameters = self.strategy.initial_parameters
        start_time = timeit.default_timer()

        # ---------------------------- get number of clients (starts) -----------------------------
        log(INFO, f"Clients available before survey: {self._client_manager.num_available()}")

        # # option 1 is an elegant idea, but currently not working
        # self.client_manager().wait_for(3, timeout=timeout)

        # option 2, work around
        self.fit_round(server_round=0, timeout=timeout)
        # TODO when eliminating option 2, be sure to adjust the client fit()

        log(INFO, f"Clients available after survey: {self._client_manager.num_available()}")
        # -------------------------- get number of clients (ends) ---------------------------------

        # assigns clients to integers > 0 as ID and sets everyone's status as online
        self.initialize_tables()

        # Perform federated learning rounds under the Secure Aggregation Protocol.
        for current_round in range(1, 1 + num_rounds):

            metrics = self.secure_aggregation(current_round, timeout)

            # Record distributed (and not centralized) loss / metrics.
            history = History()
            history.add_metrics_distributed_fit(server_round=current_round, metrics=metrics)

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
        log(INFO, "FL finished in %s", elapsed)

        if self.wandb_reporter:
            self.wandb_reporter.report_metrics(num_rounds, history)
        return history

    def secure_aggregation(self, current_round, timeout) -> Any:
        """Returns metrics"""
        self.setup(current_round)
        self.advertise_keys(current_round)
        self.share_keys(current_round)

        params, metrics, responded_clients, dropped_clients = self.masked_input_collection(
            current_round=current_round, timeout=timeout, arithmetic_modulus=self.crypto.arithmetic_modulus
        )

        self.parameters = params
        self.parameter_exchanger.pull_parameters(
            parameters=parameters_to_ndarrays(self.parameters), model=self.server_model
        )

        # server procedure to undiscretize aggregate
        model_vector = vectorize_model(self.server_model)
        half_m = self.crypto.arithmetic_modulus / 2
        for i in range(model_vector.numel()):
            if model_vector[i] > 0:
                model_vector[i] = half_m - model_vector[i]
            else:
                model_vector[i] = -model_vector[i] - half_m

        sign_diagonal_matrix = generate_sign_diagonal_matrix(self.crypto.model_dimension)
        exponent = torch.ceil(torch.log2(self.crypto.model_dimension))
        # the matrix is its inverse
        inverse_welsh_hadamard = generate_walsh_hadamard_matrix(exponent)

        # TODO matke the matrix multiplication more efficient (i.e. turn into matrix-vector mult)
        undiscretized_vector = self.privacy_settings["granularity"] * sign_diagonal_matrix
        undiscretized_vector = torch.matmul(undiscretized_vector, inverse_welsh_hadamard)

        if self.dropout_mode:
            unmaksed_params = self.unmasking(
                round=current_round, timeout=timeout, responded=responded_clients, dropped=dropped_clients
            )

        return metrics

    def initialize_tables(self):

        online_clients: List[ClientProxy] = []
        cids: List[str] = []

        for proxy in self._client_manager.all().values():
            cids.append(proxy.cid)
            online_clients.append(proxy)

        # shuffle clients for security
        permutation = np.random.permutation(len(online_clients))

        proxy_table = {}
        status_table = {}
        cid_table = {}

        for i in permutation:

            j = i.item()
            client_integer = 1 + j
            cid: str = cids[j]

            proxy_table[client_integer] = online_clients[j]
            status_table[client_integer] = Status(dropout=False)
            cid_table[cid] = client_integer

        self.id_proxy_table = proxy_table
        self.id_status_table = status_table
        self.cid_to_id_table = cid_table

    def setup(self, round: int) -> None:

        self.set_status(round=round, secagg_stage="setup")
        peers = self.get_peer_number()
        if peers < self.crypto.shamir_reconstruction_threshold:
            log(WARN, f"Too many dropouts, quitting FL.")
            exit()
        self.crypto.set_number_of_bobs(peer_count=peers)

        # Set modulus after counting peers.
        self.crypto.calculate_arithmetic_modulus()

        self.blackbox.record_setup(round, peers + 1, self.crypto.model_integer_range)

    def advertise_keys(self, round: int) -> None:
        """
        Round 0 of the Secure Aggregation Protocol.
        """

        # ---------------------------- parameterize clients (starts) -----------------------------

        metadata = {
            "current_fl_round": round,
            "number_of_bobs": self.crypto.number_of_bobs,  # peer count
            "arithmetic_modulus": self.crypto.arithmetic_modulus,
            "shamir_reconstruction_threshold": self.crypto.shamir_reconstruction_threshold,
            "dropout_mode": self.dropout_mode,
        }

        all_requests = []
        avail_clients = set()
        for id, proxy in self.id_proxy_table.items():

            if self.id_status_table[id].dropout:
                continue
            avail_clients.add(id)

            req_dict = {
                **metadata,
                "client_integer": id,
            }

            request = self.strategy.package_single_client_request(
                client=proxy, request=req_dict, event_name=Event.ADVERTISE_KEYS.value
            )
            all_requests.append(request)

        # broadcast
        online, dropouts = poll_clients(
            client_instructions=all_requests, max_workers=self.max_workers, timeout=self.timeout
        )

        # ---------------------------- parameterize clients (ends) -------------------------------

        # ---------------------------- record keys (starts) --------------------------------------

        peer_count = len(online) - 1
        if peer_count < self.crypto.shamir_reconstruction_threshold:
            log(WARN, f"Too many dropouts for SecAgg to continue, aborting FL.")
            exit()

        online_ids = set()
        for client, response in online:
            id = self.cid_to_id_table[client.cid]
            online_ids.add(id)

            res = response.properties

            # this assertion prevents a type of silent error
            assert res["event_name"] == Event.ADVERTISE_KEYS.value

            self.crypto.append_client_public_keys(
                client_integer=res["client_integer"],
                encryption_public_key=res["public_encryption_key"],
                masking_public_key=res["public_mask_key"],
            )

        dropout_ids = avail_clients.difference(online_ids)

        for id in dropout_ids:
            self.id_status_table[id].dropout = True

        # NOTE For production use, also report name of dropped out hospitals.
        log(INFO, f"FL round {round} advertise keys stage has dropout clients {dropout_ids}.")

        # ---------------------------- record keys (ends) --------------------------------------

    def share_keys(self, round) -> None:
        """Round 1 of the Secure Aggregation Protocol.

        Server communicates with clients 2x during this round:
            1. send to Alice public keys for encrypting her Shamir shares to send back
            2. broadcast each of Alice's encrypted shares to the appropriate Bob (peer client)
        """

        # ---------------------- first round (start) ----------------------------
        req = {"current_fl_round": round, "bobs_public_keys": pickle.dumps(self.crypto.get_all_public_keys())}

        # API records dropout and checks Shamir threshold by default
        res: List[Dict[str, Scalar]] = self.api(request_dict=req, event_name=Event.SHARE_KEYS.value, round=round)
        # ---------------------- first round (ends) ----------------------------

        # ---------------------- second round (start) --------------------------

        shamir_delivery = {i: {} for i in range(1, 1 + len(res))}
        """The expected structure of shamir_delivery is
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

        # NOTE clients receive this broadcast as the Event.MASKED_INPUT_COLLECTION stage. Response not stored.
        self.api_custom_message(
            custom_messages=shamir_delivery, event_name=Event.MASKED_INPUT_COLLECTION.value, round=round
        )

        # ---------------------- second round (ends) --------------------------

    def masked_input_collection(self, current_round, timeout, arithmetic_modulus):
        params, metircs, responded_and_dropouts = self.secure_aggregation_fit_round(
            server_round=current_round, timeout=timeout, arithmetic_modulus=arithmetic_modulus
        )

        # record dropouts
        online_clients = self.get_online_clients()
        responded_clients = set()
        responded, _ = responded_and_dropouts
        for proxy, _ in responded:
            id = self.cid_to_id_table[proxy.cid]
            responded_clients.add(id)
        dropouts = online_clients.difference(responded_clients)
        for client in dropouts:
            self.set_dropout(client)

        # verify Shamir threshold is maintained
        peer_count = len(self.get_online_clients()) - 1
        t = self.crypto.shamir_reconstruction_threshold
        if peer_count < t:
            error_msg = f"""Too many dropouts on round {current_round} during Masked Input Collection:
                online peers {peer_count} < {t} threshold, aborting FL."""
            log(WARN, error_msg)
            exit()

        dropped_clients = self.get_dropout_clients()
        return params, metircs, responded_clients, dropped_clients

    def unmasking(self, round: int, responded: List[ClientId], dropped: List[ClientId], timeout: float):
        """Outputs sum of masks to be removed (subtracted).

        responded: clients who contributed to model_vect during FedAvg
        dropped: clients who did not contribut to model_vect, but whose pairwise masks have been included
        """

        # if bob has dropped out, the pair mask Shamir secret will be shared by alice
        # if bob is online, the self mask Shamir secret will be shared by alice

        online_clients: Set[ClientId] = self.get_online_clients()

        message = GetPropertiesIns(
            {
                "pickled_dropout_clients": pickle.dumps(dropped),
                "pickled_online_clients": pickle.dumps(online_clients),
                "event_name": Event.UNMASKING.value,
                "current_fl_round": round,
            }
        )

        # only send message to online clients
        requests: List[Tuple[ClientProxy, GetPropertiesIns]] = [
            (self.id_proxy_table[id], message) for id in self.id_proxy_table.keys() if id not in dropped
        ]

        online, dropouts = poll_clients(client_instructions=requests, max_workers=self.max_workers, timeout=timeout)

        # verify
        peer_count = len(online) - 1
        t = self.crypto.shamir_reconstruction_threshold
        if peer_count < t:
            log(
                WARN,
                f"Insufficient Shamir shares {peer_count} < {t} received on federated round {round} for unmasking, aborting FL.",
            )
            exit()

        # record dropouts
        responded_clients: Set[ClientId] = set()
        for proxy, _ in online:
            id = self.cid_to_id_table[proxy.cid]
            responded_clients.add(id)

        droped_after_communication: Set[ClientId] = online.difference(responded_clients)
        self.set_dropout_from_list(list(droped_after_communication))

        # -------------------------------- Shamir reconstruction (start) ------------------------------------------
        # NOTE be very careful of the sign during mask removal: they should be opposite to the masking procedure

        # maps ClientID to an array of ShamirSecret
        pair_mask_shamir_dict = {}
        self_mask_shamir_dict = {}

        responses_list = [res.properties for _, res in online]
        for client_dict in responses_list:
            # verify we have the right information
            assert client_dict["event_name"] == Event.UNMASKING.value and client_dict["current_fl_round"] == round

            # dictionary mapping ClientId to ShamirSecret.
            pairmask_shamir_secrets = pickle.loads(client_dict["pickled_pairmask_shamir_secrets"])
            for id, shamir_secret in pairmask_shamir_secrets.items():
                if id in pair_mask_shamir_dict:
                    pair_mask_shamir_dict[id].append(shamir_secret)
                    continue
                # init array
                pair_mask_shamir_dict[id] = [shamir_secret]

            # dictionary mapping ClientId to ShamirSecret.
            selfmask_shamir_secrets = pickle.loads(client_dict["selfmask_shamir_secrets"])
            for id, shamir_secret in selfmask_shamir_secrets.items():
                if id in pair_mask_shamir_dict:
                    self_mask_shamir_dict[id].append(shamir_secret)
                    continue
                # init array
                self_mask_shamir_dict[id] = [shamir_secret]

        self_mask_seeds = list(
            map(
                lambda secrets_array: self.crypto.reconstruct_self_mask_seed(secrets_array),
                self_mask_shamir_dict.values(),
            )
        )

        # maps (responded, dropout) to their mutual seed
        pair_mask_seeds: Dict[Tuple[ClientId, ClientId], bytes] = {}
        for i, j in product(responded, dropped):
            pair_mask_seeds[(i, j)] = self.crypto.reconstruct_pair_mask(
                alice_shamir_shares=pair_mask_shamir_dict[j], bob_public_key=self.crypto.client_public_keys[i].mask_key
            )
        # -------------------------------- Shamir reconstruction (end) ------------------------------------------

        # ---------------------------------------------- Unmasking (start) ----------------------------------------

        # NOTE inappropriate dtype may cause slient overflow errors
        unmasking_vector = np.zeros(self.crypto.model_dimension, dtype=np.longlong)

        # process self masks
        for seed in self_mask_seeds:
            unmasking_vector -= self.crypto.recover_mask_vector(seed)

        # process pair masks
        for pair, seed in pair_mask_seeds.items():
            responded_id, dropout_id = pair

            # the responded client needs to reverse the mask by reversing the sign.
            # NOTE it is important to get the sign correct here
            if responded_id < dropout_id:
                unmasking_vector += self.crypto.recover_mask_vector(seed)
            elif responded_id > dropout_id:
                unmasking_vector -= self.crypto.recover_mask_vector(seed)
            else:
                log(WARN, "The responded ID should not equal dropout ID, something is wrong, aborting FL.")
                exit()

        unmasked_vector = torch.from_numpy(unmasking_vector) + vectorize_model(model=self.server_model)

        self.server_model = unvectorize_model(model=self.server_model, parameter_vector=unmasked_vector)
        self.parameters = ndarrays_to_parameters(
            [layer.cpu().numpy() for layer in self.server_model.state_dict().values()]
        )

        # ---------------------------------------------- Unmasking (end) ----------------------------------------

    def api(
        self,
        request_dict: Dict[str, Scalar],
        event_name: str,
        track_clients=True,
        round=None,
        verify_shamir_threshold=True,
    ) -> List[Dict[str, Scalar]]:
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

        request = self.strategy.package_request(request_dict, event_name, self._client_manager)

        log(INFO, f"API calling all clients for {event_name}")
        online, dropouts = poll_clients(
            client_instructions=request, max_workers=self.max_workers, timeout=self.timeout
        )

        if verify_shamir_threshold:
            peer_count = len(online) - 1
            t = self.crypto.shamir_reconstruction_threshold
            if peer_count < t:
                log(WARN, f"Too many dropouts: online peers {peer_count} < {t} threshold, aborting FL.")
                exit()

        responses = []

        # resolve responses
        if track_clients:

            avail_clients = self.get_online_clients()
            online_clients = set()

            for client_proxy, client_response in online:
                res = client_response.properties

                # check clients have responded to correct event
                assert res["event_name"] == event_name

                responses.append(res)

                id = self.cid_to_id_table[client_proxy.cid]
                online_clients.add(id)

            dropout_clients = avail_clients.difference(online_clients)

            for id in dropout_clients:
                self.set_dropout(id)

            if len(dropout_clients) > 0:
                log(WARN, f"{len(dropout_clients)} clients has dropped out during round {round}, {event_name}")
            else:
                log(INFO, f"No dropout during round {round}, {event_name}.")

            return responses

        # elif no client tracking
        for client_proxy, res in online:

            data = res.properties
            responses.append(data)

        return responses

    def api_custom_message(
        self,
        custom_messages: Dict[ClientId, Dict],
        event_name: str,
        track_clients=True,
        round=None,
        verify_shamir_threshold=True,
    ) -> List[Dict[str, Scalar]]:
        """This API allows the server to send messages customized for each client.
        Be careful not to deeply nest the inner dictionary as it can cause slient errors: if need be, pickle inner dictionary!
        """

        log(INFO, f"Custom API calling all clients for {event_name}")

        requests: List[Tuple[ClientProxy, GetPropertiesIns]] = []

        for client_integer, client_dict in custom_messages.items():

            """
            NOTE expected structure
            client_dict = {
                ShamirOwnerId: {
                    'encrypted_shamir_pairwise': bytes,
                    'encrypted_shamir_self': bytes
            }
            """

            client_proxy = self.id_proxy_table[client_integer]
            client_req = GetPropertiesIns({"event_name": event_name, "pickled_message": pickle.dumps(client_dict)})

            # NOTE we append a tuple
            requests.append((client_proxy, client_req))

        online, dropouts = poll_clients(
            client_instructions=requests, max_workers=self.max_workers, timeout=self.timeout
        )

        if verify_shamir_threshold:
            peer_count = len(online) - 1
            t = self.crypto.shamir_reconstruction_threshold
            if peer_count < t:
                log(WARN, f"Too many dropouts: online peers {peer_count} < {t} threshold, aborting FL.")
                exit()

        responses = []

        # resolve responses
        if track_clients:

            avail_clients = self.get_online_clients()
            online_clients = set()

            for client_proxy, client_response in online:
                res = client_response.properties

                # check clients have responded to correct event
                assert res["event_name"] == event_name

                responses.append(res)

                id = self.cid_to_id_table[client_proxy.cid]
                online_clients.add(id)

            dropout_clients = avail_clients.difference(online_clients)

            for id in dropout_clients:
                self.set_dropout(id)

            if len(dropout_clients) > 0:
                log(WARN, f"{len(dropout_clients)} clients has dropped out during round {round}, {event_name}")
            else:
                log(INFO, f"No dropout during round {round}, {event_name}.")

            return responses

    def secure_aggregation_fit_round(
        self, server_round: int, timeout: Optional[float], arithmetic_modulus: int
    ) -> Optional[Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]]:
        """Performs one FedAvg round with modular arithmetic and post processing for secure aggregation.

        Reference
        Algorithm 2 "The Distributed Discrete Gaussian Mechanism for Federated Learning with Secure Aggregation"
        https://arxiv.org/pdf/2102.06387.pdf

        NOTE This is a customization of fit_round in flwr.
        """
        assert isinstance(self.strategy, SecureAggregationStrategy)

        # broadcast global model parameters to clients
        client_instructions = self.strategy.configure_fit(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self._client_manager,
        )
        if not client_instructions:
            log(WARN, f"SecAgg fit round {server_round}: no clients selected, aborting.")
            exit()

        all_online_clients = self.proxys_to_ids(proxy_list=self._client_manager.all().values())

        # Get 100% clients for SecAgg (SecAgg+ samples < 100%)
        sample_count = len(client_instructions)
        assert sample_count == len(all_online_clients)
        log(INFO, f"SecAgg fit round {server_round}: strategy sampled {sample_count} clients.")

        results, failures = fit_clients(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
        )

        peer_count = self.get_peer_number()
        t = self.crypto.shamir_reconstruction_threshold
        if peer_count < t:
            error_message = f"""
                Too many dropouts on round {round} during Masked Input Collection:
                online peers {peer_count} < {t} threshold, aborting FL.
            """
            log(WARN, error_message)
            exit()

        responded = self.proxys_to_ids(proxy_list=[proxy for proxy, _ in results])

        # to be returned
        dropouts = list(all_online_clients.difference(responded))
        self.set_dropout_from_list(dropouts)

        log(
            WARN,
            "fit_round %s received %s results and %s failures",
            server_round,
            len(results),
            len(failures),
        )

        # Aggregate training results
        aggregated_result: Tuple[Optional[Parameters], Dict[str, Scalar],] = self.strategy.aggregate_fit(
            server_round=server_round,
            arithmetic_modulus=self.crypto.arithmetic_modulus,
            results=results,
            failures=failures,
        )

        parameters_aggregated, metrics_aggregated = aggregated_result
        return parameters_aggregated, metrics_aggregated, (results, failures)

    def proxys_to_ids(self, proxy_list: List[ClientProxy]) -> Set[ClientId]:
        ids = set()
        for proxy in proxy_list:
            id = self.cid_to_id_table[proxy.cid]
            ids.add(id)
        return ids

    def get_dropout_clients(self) -> List[int]:
        dropouts = []
        for id, status in self.id_status_table.items():
            if status.dropout:
                dropouts.append(id)
        return dropouts

    def get_online_clients(self) -> Set[int]:
        """Set of Client IDs who have not dropped since FL started."""
        avail_clients = set()
        for id, status in self.id_status_table.items():
            if not status.dropout:
                avail_clients.add(id)
        return avail_clients

    def set_shamir_threshold(self, threshold: int) -> None:
        assert isinstance(threshold, int) and threshold >= 2
        if self.crypto is None:
            log(INFO, f"Must initialize ServerCryptoKit before setting Shamir threshold: self.crypto is None")
            return
        self.crypto.set_shamir_threshold(threshold)
        log(INFO, f"Shamir reconstruction threshold set to {threshold}")

    def set_model_integer_range(self, model_integer_range: int) -> None:
        assert isinstance(model_integer_range, int) and model_integer_range > 1
        if self.crypto is None:
            log(INFO, f"Must initialize ServerCryptoKit before setting model_integer_range: self.crypto is None")
            return
        self.crypto.set_model_integer_range(model_integer_range)
        log(INFO, f"Model integer range set to {model_integer_range}")

    def get_arithmetic_modulus(self) -> int:
        return self.crypto.calculate_arithmetic_modulus()

    def set_status(self, round: int, secagg_stage: str) -> None:
        # record dropouts
        online_clients = set()
        for proxy in self._client_manager.all().values():
            online_clients.add(proxy.cid)
        all_clients = set(self.cid_to_id_table.keys())
        dropouts = all_clients.difference(online_clients)
        for cid in dropouts:
            id = self.cid_to_id_table[cid]
            self.id_status_table[id].dropout = True
        if len(dropouts) > 0:
            log(WARN, f"FL round {round}, {secagg_stage} phase observed {len(dropouts)} dropouts: {dropouts}")
        else:
            log(INFO, f"FL round {round}, {secagg_stage} observed no dropout.")

    def set_online(self, client_integer: ClientId) -> None:
        assert client_integer in self.id_status_table
        self.id_status_table[client_integer].dropout = False

    def set_dropout(self, client_integer: ClientId) -> None:
        assert client_integer in self.id_status_table
        self.id_status_table[client_integer].dropout = True

    def set_dropout_from_list(self, dropout_list: List[ClientId]) -> None:
        for id in dropout_list:
            self.set_dropout(id)

    def get_peer_number(self):
        """The number of peers of Alice (total number of clients - 1)"""
        return self._client_manager.num_available() - 1

    def get_metadata(self, current_fl_round) -> Dict[str, Scalar]:

        return {
            "sender": "server",
            "current_fl_round": current_fl_round,
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
