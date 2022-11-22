from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Union

from dp_accounting import (
    DpEvent,
    DpEventBuilder,
    GaussianDpEvent,
    PoissonSampledDpEvent,
    SampledWithoutReplacementDpEvent,
    SelfComposedDpEvent,
)
from dp_accounting.rdp.rdp_privacy_accountant import NeighborRel, RdpAccountant


class SamplingStrategy(ABC):
    def __init__(self, neighbor_relation: NeighborRel) -> None:
        self.neighbor_relation = neighbor_relation

    @abstractmethod
    def get_dp_event(self, noise_event: DpEvent) -> DpEvent:
        raise NotImplementedError


class PoissonSampling(SamplingStrategy):
    def __init__(self, sampling_ratio: float) -> None:
        self.sampling_ratio = sampling_ratio
        super().__init__(NeighborRel.ADD_OR_REMOVE_ONE)

    def get_dp_event(self, noise_event: DpEvent) -> DpEvent:
        return PoissonSampledDpEvent(self.sampling_ratio, noise_event)


class FixedSamplingWithoutReplacement(SamplingStrategy):
    def __init__(self, population_size: int, sample_size: int) -> None:
        self.population_size = population_size
        self.sample_size = sample_size
        super().__init__(NeighborRel.REPLACE_ONE)

    def get_dp_event(self, noise_event: DpEvent) -> DpEvent:
        return SampledWithoutReplacementDpEvent(self.population_size, self.sample_size, noise_event)


class MomentsAccountant:
    def __init__(self, moment_orders: Optional[List[float]] = None) -> None:
        """Moment orders are equivalent to lambda from Deep Learning with Differential Privacy (Abadi et. al. 2016).
        They form the set of moments to estimate the infimum of Theorem 2 part 2. The default values were taken from
        the tensorflow federated DP tutorial notebook:
        https://github.com/tensorflow/federated/blob/main/docs/tutorials/
        federated_learning_with_differential_privacy.ipynb

        In the paper above, they state that trying lambda <= 32 is usually sufficient.

        Sampling type is the data point sampling strategy: i.e. examples from dataset for a batch with probability q
        Noise type specifies whether Gaussian or Laplacian noise is added to the updates
        """
        if moment_orders is not None:
            self.moment_orders = moment_orders
        else:
            low_orders = [1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 3.5, 4.0, 4.5]
            medium_orders: List[float] = list(range(5, 64))
            high_orders = [128.0, 256.0, 512.0]
            self.moment_orders = low_orders + medium_orders + high_orders

    def _construct_dp_events(
        self, sampling_strategy: SamplingStrategy, noise_multiplier: float, updates: int
    ) -> DpEvent:
        # Type of noise used for DP on gradient updates
        noise_event = GaussianDpEvent(noise_multiplier)
        # Type of strategy used to select datapoints (or clients in user-level FL DP)
        sampling_event = sampling_strategy.get_dp_event(noise_event)
        # Number of times the above two procedures are performed (e.g epochs*batches for DP-SGD)
        return SelfComposedDpEvent(sampling_event, updates)

    def _construct_dp_events_trajectory(
        self,
        sampling_strategies: Sequence[SamplingStrategy],
        noise_multipliers: List[float],
        updates_list: List[int],
    ) -> DpEvent:
        # Given a list of parameters this assumes that the DP operations were performed in sequence
        event_builder = DpEventBuilder()
        for sampling_strategy, noise_multiplier, updates in zip(sampling_strategies, noise_multipliers, updates_list):
            event_builder.compose(self._construct_dp_events(sampling_strategy, noise_multiplier, updates), 1)
        return event_builder.build()

    def _construct_rdp_accountant(
        self,
        sampling_strategies: Union[SamplingStrategy, Sequence[SamplingStrategy]],
        noise_multipliers: Union[float, List[float]],
        updates: Union[int, List[int]],
    ) -> RdpAccountant:
        if isinstance(sampling_strategies, SamplingStrategy):
            sampling_strategies = [sampling_strategies]
        if isinstance(noise_multipliers, float):
            noise_multipliers = [noise_multipliers]
        if isinstance(updates, int):
            updates = [updates]

        # First we construct the DP events for the accountant to accumulate
        dp_events = self._construct_dp_events_trajectory(sampling_strategies, noise_multipliers, updates)
        # Setup accountant with set of moments to minimize over
        rdp_accountant = RdpAccountant(self.moment_orders, sampling_strategies[0].neighbor_relation)
        # Set accountant internal state about applied events
        rdp_accountant.compose(dp_events)
        return rdp_accountant

    def _validate_accountant_input(
        self,
        sampling_strategies: Union[SamplingStrategy, Sequence[SamplingStrategy]],
        noise_multiplier: Union[float, List[float]],
        updates: Union[int, List[int]],
    ) -> None:
        all_lists = all(
            [
                isinstance(sampling_strategies, Sequence)
                and isinstance(noise_multiplier, list)
                and isinstance(updates, list)
            ]
        )
        all_values = all(
            [
                isinstance(sampling_strategies, SamplingStrategy)
                and isinstance(noise_multiplier, float)
                and isinstance(updates, int)
            ]
        )
        assert all_lists or all_values

    def get_epsilon(
        self,
        sampling_strategies: Union[SamplingStrategy, Sequence[SamplingStrategy]],
        noise_multiplier: Union[float, List[float]],
        updates: Union[int, List[int]],
        delta: float,
    ) -> float:
        """
        If the parameters are lists, then it is assumed that the training applied the parameters in a sequence of
        updates.
            Ex. sampling_strategies = [PoissonSampling(q_1), PoissonSampling(q_2)], noise_multiplier = [z_1, z_2],
            updates = [t_1, t_2] implies that q_1, z_1 were applied for t_1 updates, followed by q_2, z_2 for t_2
            updates.
        Sampling_strategies: Are the type of sampling done for each datapoint or client in the DP procedure.
            This is either Poisson sampling with a sampling rate specified or Fixed ratio sampling with a fixed number
            of selections performed over a specified population size.
            For non-FL DP-SGD: This is the ratio of batch size to dataset size
            (L/N, from Deep Learning with Differential Privacy).
            For FL with clientside DP-SGD (no noise on server side, instance level privacy): This is the ratio of
            client sampling probability to client data point probability q*(b_k/n_k)
            For FL with client privacy: This is the sampling of clients from the client population
            NOTE: If a sequence of strategies is given, they must be all of the same kind (that is poisson or subset,
            but may have different parameters)
         Noise multiplier: Ratio of the noise standard deviation to clipping bound (sigma in Deep Learning with
            Differential Privacy, z in some other implementations).
         Updates: This is the number of noise applications to the update weights.
            For non-FL DP-SGD: This is the number of updates run (epochs*batches per epoch)
            For FL w/ clientside DP-SGD (instance DP): This is the number of batches run per client (if selected
            everytime), server_updates*epochs per server update*batches per epoch
            For FL with client privacy: Number of server updates
         Delta: This is the delta in (epsilon, delta)-Privacy, that we require."""
        self._validate_accountant_input(sampling_strategies, noise_multiplier, updates)
        rdp_accountant = self._construct_rdp_accountant(sampling_strategies, noise_multiplier, updates)
        # calculate minimum epsilon for fixed delta
        return rdp_accountant.get_epsilon(delta)

    def get_delta(
        self,
        sampling_strategies: Union[SamplingStrategy, Sequence[SamplingStrategy]],
        noise_multiplier: Union[float, List[float]],
        updates: Union[int, List[int]],
        epsilon: float,
    ) -> float:
        """
        If the parameters are lists, then it is assumed that the training applied the parameters in a sequence of
        updates.
            Ex. sampling_strategies = [PoissonSampling(q_1), PoissonSampling(q_2)], noise_multiplier = [z_1, z_2],
            updates = [t_1, t_2] implies that q_1, z_1 were applied for t_1 updates, followed by q_2, z_2 for t_2
            updates.
        Sampling_strategies: Are the type of sampling done for each datapoint or client in the DP procedure.
            This is either Poisson sampling with a sampling rate specified or Fixed ratio sampling with a fixed number
            of selections performed over a specified population size.
            For non-FL DP-SGD: This is the ratio of batch size to dataset size
            (L/N, from Deep Learning with Differential Privacy).
            For FL with clientside DP-SGD (no noise on server side, instance level privacy): This is the ratio of
            client sampling probability to client data point probability q*(b_k/n_k)
            For FL with client privacy: This is the sampling of clients from the client population
            NOTE: If a sequence of strategies is given, they must be all of the same kind (that is poisson or subset,
            but may have different parameters)
         Noise multiplier: Ratio of the noise standard deviation to clipping bound (sigma in Deep Learning with
            Differential Privacy, z in some other implementations).
         Updates: This is the number of noise applications to the update weights.
            For non-FL DP-SGD: This is the number of updates run (epochs*batches per epoch)
            For FL w/ clientside DP-SGD (instance DP): This is the number of batches run per client (if selected
            everytime), server_updates*epochs per server update*batches per epoch
            For FL with client privacy: Number of server updates
         epsilon: This is the epsilon in (epsilon, delta)-Privacy, that we require."""
        self._validate_accountant_input(sampling_strategies, noise_multiplier, updates)
        rdp_accountant = self._construct_rdp_accountant(sampling_strategies, noise_multiplier, updates)
        # calculate minimum delta for fixed epsilon
        return rdp_accountant.get_delta(epsilon)
