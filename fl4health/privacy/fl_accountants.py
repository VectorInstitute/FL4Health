from abc import ABC, abstractmethod
from math import ceil
from typing import List, Optional, Union

from fl4health.privacy.moments_accountant import (
    FixedSamplingWithoutReplacement,
    MomentsAccountant,
    PoissonSampling,
    SamplingStrategy,
)


class FlInstanceLevelAccountant:
    """
    This accountant should be used when applying FL and measuring instance-level privacy
    NOTE: This class assumes that all sampling is done via Poisson sampling (client and data point level).
    Further it assumes that the sampling ratio of clients and noise multiplier are fixed throughout training
    """

    def __init__(
        self,
        client_sampling_rate: float,
        noise_multiplier: float,
        epochs_per_round: int,
        client_batch_sizes: List[int],
        client_dataset_sizes: List[int],
        moment_orders: Optional[List[float]] = None,
    ) -> None:
        """
        client_sampling_rate: probability that each client will be included in a round
        noise_multiplier: multiplier of noise std. dev. on clipping bound
        epochs_per_round: number of epochs each client will complete per server round
        client_batch_sizes: batch size per client, if a single value it is assumed to be constant across clients
        client_dataset_sizes: size of full dataset on a client, if a single value it is assumed to be constant
        across clients.
        """
        self.noise_multiplier = noise_multiplier
        self.epochs_per_round = epochs_per_round
        assert len(client_batch_sizes) == len(client_dataset_sizes)

        self.num_batches_per_client = self._calculate_num_batches(client_batch_sizes, client_dataset_sizes)

        client_batch_ratios = self._calculate_batch_ratios(client_batch_sizes, client_dataset_sizes)
        self.sampling_strategies_per_client = [
            PoissonSampling(client_sampling_rate * client_batch_ratio) for client_batch_ratio in client_batch_ratios
        ]

        self.accountant = MomentsAccountant(moment_orders)

    def _calculate_batch_ratios(self, client_batch_sizes: List[int], client_dataset_sizes: List[int]) -> List[float]:
        return [batch / dataset for batch, dataset in zip(client_batch_sizes, client_dataset_sizes)]

    def _calculate_num_batches(self, client_batch_sizes: List[int], client_dataset_sizes: List[int]) -> List[int]:
        return [ceil(dataset / batch) for batch, dataset in zip(client_batch_sizes, client_dataset_sizes)]

    def get_epsilon(self, server_updates: int, delta: float) -> float:
        """server_updates: number of central server updates performed"""
        epsilons = []
        for num_batch, sampling_strategy in zip(self.num_batches_per_client, self.sampling_strategies_per_client):
            total_updates = server_updates * self.epochs_per_round * num_batch
            epsilon = self.accountant.get_epsilon(sampling_strategy, self.noise_multiplier, total_updates, delta)
            epsilons.append(epsilon)
        return max(epsilons)

    def get_delta(self, server_updates: int, epsilon: float) -> float:
        """server_updates: number of central server updates performed"""
        deltas = []
        for num_batch, sampling_strategy in zip(self.num_batches_per_client, self.sampling_strategies_per_client):
            total_updates = server_updates * self.epochs_per_round * num_batch
            delta = self.accountant.get_delta(sampling_strategy, self.noise_multiplier, total_updates, epsilon)
            deltas.append(delta)
        return max(deltas)


class ClientLevelAccountant(ABC):
    def __init__(
        self, noise_multiplier: Union[float, List[float]], moment_orders: Optional[List[float]] = None
    ) -> None:
        self.noise_multiplier = noise_multiplier
        self.accountant = MomentsAccountant(moment_orders)

    @abstractmethod
    def get_epsilon(self, server_updates: Union[int, List[int]], delta: float) -> float:
        pass

    @abstractmethod
    def get_delta(self, server_updates: Union[int, List[int]], epsilon: float) -> float:
        pass

    def _validate_server_updates(self, server_updates: Union[int, List[int]]) -> None:
        if isinstance(server_updates, list):
            assert isinstance(self.noise_multiplier, list)
            assert len(server_updates) == len(self.noise_multiplier)
        else:
            assert isinstance(self.noise_multiplier, float)


class FlClientLevelAccountantPoissonSampling(ClientLevelAccountant):
    """
    This accountant should be used when applying FL with Poisson client sampling and measuring client-level privacy
    """

    def __init__(
        self,
        client_sampling_rate: Union[float, List[float]],
        noise_multiplier: Union[float, List[float]],
        moment_orders: Optional[List[float]] = None,
    ) -> None:
        """
        client_sampling_rate: probability that each client will be included in a round
        noise_multiplier: multiplier of noise std. dev. on clipping bound
        NOTE: The above values can be lists, where they are treated as sequences of training with the respective
        parameters
        """
        super().__init__(noise_multiplier, moment_orders)
        self.sampling_strategy: Union[SamplingStrategy, List[PoissonSampling]]

        if isinstance(client_sampling_rate, list):
            self.sampling_strategy = [PoissonSampling(q) for q in client_sampling_rate]
        else:
            self.sampling_strategy = PoissonSampling(client_sampling_rate)

    def get_epsilon(self, server_updates: Union[int, List[int]], delta: float) -> float:
        """server_updates: number of central server updates performed"""
        self._validate_server_updates(server_updates)
        return self.accountant.get_epsilon(self.sampling_strategy, self.noise_multiplier, server_updates, delta)

    def get_delta(self, server_updates: Union[int, List[int]], epsilon: float) -> float:
        """server_updates: number of central server updates performed"""
        self._validate_server_updates(server_updates)
        return self.accountant.get_delta(self.sampling_strategy, self.noise_multiplier, server_updates, epsilon)


class FlClientLevelAccountantFixedSamplingNoReplacement(ClientLevelAccountant):
    """
    This accountant should be used when applying FL with Poisson client sampling and measuring client-level privacy
    """

    def __init__(
        self,
        n_total_clients: int,
        n_clients_sampled: Union[int, List[int]],
        noise_multiplier: Union[float, List[float]],
        moment_orders: Optional[List[float]] = None,
    ) -> None:
        """
        n_total_clients: total number of clients to be sampled from
        n_clients_sampled: number of clients sampled in a given round
        noise_multiplier: multiplier of noise std. dev. on clipping bound
        NOTE: The above values can be lists, where they are treated as sequences of training with the respective
        parameters
        """
        super().__init__(noise_multiplier, moment_orders)
        self.sampling_strategy: Union[SamplingStrategy, List[FixedSamplingWithoutReplacement]]

        if isinstance(n_clients_sampled, list):
            self.sampling_strategy = [
                FixedSamplingWithoutReplacement(n_total_clients, n_clients) for n_clients in n_clients_sampled
            ]
        else:
            self.sampling_strategy = FixedSamplingWithoutReplacement(n_total_clients, n_clients_sampled)

    def get_epsilon(self, server_updates: Union[int, List[int]], delta: float) -> float:
        """server_updates: number of central server updates performed"""
        self._validate_server_updates(server_updates)
        return self.accountant.get_epsilon(self.sampling_strategy, self.noise_multiplier, server_updates, delta)

    def get_delta(self, server_updates: Union[int, List[int]], epsilon: float) -> float:
        """server_updates: number of central server updates performed"""
        self._validate_server_updates(server_updates)
        return self.accountant.get_delta(self.sampling_strategy, self.noise_multiplier, server_updates, epsilon)
