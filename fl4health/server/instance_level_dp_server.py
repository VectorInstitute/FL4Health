from logging import INFO
from typing import List, Optional

from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.history import History

from fl4health.privacy.fl_accountants import FlInstanceLevelAccountant
from fl4health.server.base_server import FlServer
from fl4health.server.polling import poll_clients
from fl4health.strategies.instance_level_dp_fedavg import InstanceLevelDPFedAvgSampling


class InstanceLevelDPServer(FlServer):
    """
    Server to be used in case of Instance Level Differential Privacy with Federated Averaging.
    Modified the fit function to poll clients for sample counts prior to the first round of FL.
    """

    def __init__(
        self,
        *,
        client_manager: ClientManager,
        strategy: InstanceLevelDPFedAvgSampling,
        noise_multiplier: int,
        local_epochs: int,
        batch_size: int,
        num_server_rounds: int,
        delta: Optional[float] = None,
    ) -> None:
        super().__init__(client_manager=client_manager, strategy=strategy)
        self.noise_multiplier = noise_multiplier
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.num_server_rounds = num_server_rounds
        self.delta = delta

    def fit(self, num_rounds: int, timeout: Optional[float]) -> History:
        """Run federated averaging for a number of rounds."""

        # Poll clients for sample counts
        log(INFO, "Polling Clients for sample counts")
        assert isinstance(self.strategy, InstanceLevelDPFedAvgSampling)
        client_instructions = self.strategy.configure_poll(server_round=0, client_manager=self._client_manager)
        results, _ = poll_clients(
            client_instructions=client_instructions, max_workers=self.max_workers, timeout=timeout
        )

        sample_counts: List[int] = [
            int(get_properties_res.properties["num_samples"]) for (_, get_properties_res) in results
        ]

        self.setup_privacy_accountant(sample_counts)

        return super().fit(num_rounds=num_rounds, timeout=timeout)

    def setup_privacy_accountant(self, sample_counts: List[int]) -> None:
        assert isinstance(self.strategy, InstanceLevelDPFedAvgSampling)
        self.accountant = FlInstanceLevelAccountant(
            client_sampling_rate=self.strategy.fraction_fit,
            noise_multiplier=self.noise_multiplier,
            epochs_per_round=self.local_epochs,
            client_batch_sizes=[self.batch_size for _ in range(len(sample_counts))],
            client_dataset_sizes=sample_counts,
        )

        total_samples = sum(sample_counts)
        target_delta = 1.0 / total_samples if self.delta is None else self.delta
        epsilon = self.accountant.get_epsilon(self.num_server_rounds, target_delta)
        log(INFO, f"Model privacy after full training will be ({epsilon}, {target_delta})")
