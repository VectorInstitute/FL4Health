from abc import ABC, abstractmethod
from math import ceil

from fl4health.privacy.moments_accountant import (
    FixedSamplingWithoutReplacement,
    MomentsAccountant,
    PoissonSampling,
    SamplingStrategy,
)


class FlInstanceLevelAccountant:
    def __init__(
        self,
        client_sampling_rate: float,
        noise_multiplier: float,
        epochs_per_round: int,
        client_batch_sizes: list[int],
        client_dataset_sizes: list[int],
        moment_orders: list[float] | None = None,
    ) -> None:
        """
        This accountant should be used when applying FL and measuring instance-level privacy.

        **NOTE**: This class assumes that all sampling is done via Poisson sampling (client and data point level).
        Further it assumes that the sampling ratio of clients and noise multiplier are fixed throughout training

        Args:
            client_sampling_rate (float): Probability that each client will be included in a round
            noise_multiplier (float):  Multiplier of noise std. dev. on clipping bound.
            epochs_per_round (int): Number of epochs each client will complete per server round.
            client_batch_sizes (list[int]): Batch size per client, if a single value it is assumed to be constant
                across clients.
            client_dataset_sizes (list[int]): Size of full dataset on a client, if a single value it is assumed to be
                constant across clients.
            moment_orders (list[float] | None, optional): Moments orders to be used in computing the approximate
                epsilon value. Defaults to None.
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

    def _calculate_batch_ratios(self, client_batch_sizes: list[int], client_dataset_sizes: list[int]) -> list[float]:
        return [batch / dataset for batch, dataset in zip(client_batch_sizes, client_dataset_sizes)]

    def _calculate_num_batches(self, client_batch_sizes: list[int], client_dataset_sizes: list[int]) -> list[int]:
        return [ceil(dataset / batch) for batch, dataset in zip(client_batch_sizes, client_dataset_sizes)]

    def get_epsilon(self, server_updates: int, delta: float) -> float:
        """
        Compute the epsilon value for the provided delta and the number of server updates performed.

        Args:
            server_updates (int): Number of central server updates performed.
            delta (float): Delta value from which to compute epsilon.

        Returns:
            (float): Epsilon.
        """
        epsilons = []
        for num_batch, sampling_strategy in zip(self.num_batches_per_client, self.sampling_strategies_per_client):
            # Round up because privacy loss is monotonic wrt total_updates
            total_updates = ceil(server_updates * self.epochs_per_round * num_batch)
            epsilon = self.accountant.get_epsilon(sampling_strategy, self.noise_multiplier, total_updates, delta)
            epsilons.append(epsilon)
        return max(epsilons)

    def get_delta(self, server_updates: int, epsilon: float) -> float:
        """
        Compute the delta value for the provided epsilon and the number of server updates performed.

        Args:
            server_updates (int): Number of central server updates performed.
            epsilon (float): Epsilon value from which to compute delta.

        Returns:
            (float): delta.
        """
        deltas = []
        for num_batch, sampling_strategy in zip(self.num_batches_per_client, self.sampling_strategies_per_client):
            # Round up because privacy loss is monotonic wrt total_updates
            total_updates = ceil(server_updates * self.epochs_per_round * num_batch)
            delta = self.accountant.get_delta(sampling_strategy, self.noise_multiplier, total_updates, epsilon)
            deltas.append(delta)
        return max(deltas)


class ClientLevelAccountant(ABC):
    def __init__(self, noise_multiplier: float | list[float], moment_orders: list[float] | None = None) -> None:
        """
        Accountant to be used when measuring Client Level DP in FL training.

        Args:
            noise_multiplier (float | list[float]): The noise multiplier being applied to weights before transfer to
                the server.
            moment_orders (list[float] | None, optional): Basis orders to be used by the accountant for approximation.
                of the RDP values. Defaults to None.
        """
        self.noise_multiplier = noise_multiplier
        self.accountant = MomentsAccountant(moment_orders)

    @abstractmethod
    def get_epsilon(self, server_updates: int | list[int], delta: float) -> float:
        pass

    @abstractmethod
    def get_delta(self, server_updates: int | list[int], epsilon: float) -> float:
        pass

    def _validate_server_updates(self, server_updates: int | list[int]) -> None:
        if isinstance(server_updates, list):
            assert isinstance(self.noise_multiplier, list)
            assert len(server_updates) == len(self.noise_multiplier)
        else:
            assert isinstance(self.noise_multiplier, float)


class FlClientLevelAccountantPoissonSampling(ClientLevelAccountant):
    def __init__(
        self,
        client_sampling_rate: float | list[float],
        noise_multiplier: float | list[float],
        moment_orders: list[float] | None = None,
    ) -> None:
        """
        This accountant should be used when applying FL with Poisson client sampling and measuring client-level
        privacy.

        **NOTE**: The above values can be lists, where they are treated as sequences of training with the respective
        parameters

        Args:
            client_sampling_rate (float | list[float]): Probability that each client will be included in a round.
            noise_multiplier (float | list[float]): Multiplier of noise std. dev. on clipping bound.
            moment_orders (list[float] | None, optional): Moments orders to be used in computing the approximate
                epsilon value. Defaults to None. Defaults to None.
        """
        super().__init__(noise_multiplier, moment_orders)
        self.sampling_strategy: SamplingStrategy | list[PoissonSampling]

        if isinstance(client_sampling_rate, list):
            self.sampling_strategy = [PoissonSampling(q) for q in client_sampling_rate]
        else:
            self.sampling_strategy = PoissonSampling(client_sampling_rate)

    def get_epsilon(self, server_updates: int | list[int], delta: float) -> float:
        """
        Compute the epsilon value for the provided delta and the number of server updates performed.

        Args:
            server_updates (int | list[int]): Number of central server updates performed.
            delta (float): Delta value from which to compute epsilon.

        Returns:
            (float): epsilon.
        """
        self._validate_server_updates(server_updates)
        return self.accountant.get_epsilon(self.sampling_strategy, self.noise_multiplier, server_updates, delta)

    def get_delta(self, server_updates: int | list[int], epsilon: float) -> float:
        """
        Compute the delta value for the provided epsilon and the number of server updates performed.

        Args:
            server_updates (int | list[int]): Number of central server updates performed.
            epsilon (float): Epsilon value from which to compute delta.

        Returns:
            (float): delta.
        """
        self._validate_server_updates(server_updates)
        return self.accountant.get_delta(self.sampling_strategy, self.noise_multiplier, server_updates, epsilon)


class FlClientLevelAccountantFixedSamplingNoReplacement(ClientLevelAccountant):
    def __init__(
        self,
        n_total_clients: int,
        n_clients_sampled: int | list[int],
        noise_multiplier: float | list[float],
        moment_orders: list[float] | None = None,
    ) -> None:
        """
        This accountant should be used when applying FL with Fixed Sampling with No Replacement and measuring
        client-level privacy.

        **NOTE**: The above values can be lists, where they are treated as sequences of training with the respective
        parameters

        Args:
            n_total_clients (int): Total number of clients to be sampled from.
            n_clients_sampled (int | list[int]): Number of clients sampled in a given round.
            noise_multiplier (float | list[float]): Multiplier of noise std. dev. on clipping bound.
            moment_orders (list[float] | None, optional): Moments orders to be used in computing the approximate
                epsilon value. Defaults to None. Defaults to None.
        """
        super().__init__(noise_multiplier, moment_orders)
        self.sampling_strategy: SamplingStrategy | list[FixedSamplingWithoutReplacement]

        if isinstance(n_clients_sampled, list):
            self.sampling_strategy = [
                FixedSamplingWithoutReplacement(n_total_clients, n_clients) for n_clients in n_clients_sampled
            ]
        else:
            self.sampling_strategy = FixedSamplingWithoutReplacement(n_total_clients, n_clients_sampled)

    def get_epsilon(self, server_updates: int | list[int], delta: float) -> float:
        """
        Compute the epsilon value for the provided delta and the number of server updates performed.

        Args:
            server_updates (int | list[int]): Number of central server updates performed.
            delta (float): Delta value from which to compute epsilon.

        Returns:
            (float): epsilon.
        """
        self._validate_server_updates(server_updates)
        return self.accountant.get_epsilon(self.sampling_strategy, self.noise_multiplier, server_updates, delta)

    def get_delta(self, server_updates: int | list[int], epsilon: float) -> float:
        """
        Compute the delta value for the provided epsilon and the number of server updates performed.

        Args:
            server_updates (int | list[int]): Number of central server updates performed.
            epsilon (float): Epsilon value from which to compute delta.

        Returns:
            (float): delta.
        """
        self._validate_server_updates(server_updates)
        return self.accountant.get_delta(self.sampling_strategy, self.noise_multiplier, server_updates, epsilon)
