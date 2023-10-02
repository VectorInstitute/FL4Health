from logging import DEBUG, ERROR
from typing import Optional

from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.history import History
from flwr.server.server import fit_clients

from fl4health.checkpointing.checkpointer import TorchCheckpointer
from fl4health.reporting.fl_wanb import ServerWandBReporter
from fl4health.server.base_server import FlServer
from fl4health.server.instance_level_dp_server import InstanceLevelDPServer
from fl4health.strategies.scaffold import Scaffold


class ScaffoldServer(FlServer):
    """
    Custom FL Server for scaffold algorithm to handle warm initialization of control variates
    as specified in https://arxiv.org/abs/1910.06378
    """

    def __init__(
        self,
        client_manager: ClientManager,
        strategy: Scaffold,
        wandb_reporter: Optional[ServerWandBReporter] = None,
        checkpointer: Optional[TorchCheckpointer] = None,
        warm_start: bool = False,  # Whether or not to initialize control variates of each client as local gradient
    ) -> None:
        FlServer.__init__(
            self,
            client_manager=client_manager,
            strategy=strategy,
            wandb_reporter=wandb_reporter,
            checkpointer=checkpointer,
        )
        self.warm_start = warm_start

    def initialize_paramameters(self, timeout: Optional[float]) -> None:
        """
        Initialize parameters (models weights and control variates) of the server.
        If warm_start is True, control variates are initialized as the
        the average local gradient (while model weights remain unchanged)
        """
        assert isinstance(self.strategy, Scaffold)

        self.parameters = self._get_initial_parameters(timeout=timeout)

        # If warm_start, run routine to initialize control variates without updating global model
        # control variates are initialized as average local gradient over training steps
        # while the model weights remain unchanged (until the FL rounds start)
        if self.warm_start:
            client_instructions = self.strategy.configure_fit_all(
                server_round=0, parameters=self.parameters, client_manager=self._client_manager
            )
            if not client_instructions:
                log(ERROR, "Warm Start initialization failed: no clients selected", 1)
            else:
                log(
                    DEBUG,
                    f"Warm start: strategy sampled {len(client_instructions)} \
                    clients (out of {self._client_manager.num_available()})",
                )

                results, failures = fit_clients(client_instructions, self.max_workers, timeout)

                log(
                    DEBUG,
                    f"Warm Start: received {len(results)} results and {len(failures)} failures",
                )

                updated_params = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
                aggregated_params = self.strategy.aggregate(updated_params)

                # drop the updated weights as the warm start strictly updates control variates
                # and leaves model weights unchanged
                _, control_variates_update = self.strategy.parameter_packer.unpack_parameters(aggregated_params)
                server_control_variates = self.strategy.compute_updated_control_variates(control_variates_update)

                # Get initial weights from original parameters
                initial_weights, _ = self.strategy.parameter_packer.unpack_parameters(
                    parameters_to_ndarrays(self.parameters)
                )

                # Get new parameters by combining original weights with server control variates from warm start
                self.parameters = ndarrays_to_parameters(
                    self.strategy.parameter_packer.pack_parameters(initial_weights, server_control_variates)
                )

        # Set updated initial parameters in strategy because they are deleted on every
        # self._get_initial_parameters call (there will be another one in parent fit method)
        self.strategy.initial_parameters = self.parameters

    def fit(self, num_rounds: int, timeout: Optional[float]) -> History:
        """Run Scaffold algorithm for a number of rounds."""

        assert isinstance(self.strategy, Scaffold)

        # Initialize parameters attribute (specifc to the Scaffold algo)
        self.initialize_paramameters(timeout=timeout)

        return super().fit(num_rounds=num_rounds, timeout=timeout)


class DPScaffoldServer(ScaffoldServer, InstanceLevelDPServer):
    """
    Custom FL Server for Instance Level Differentially Private Scaffold algorithm
    as specified in https://arxiv.org/abs/2111.09278
    """

    def __init__(
        self,
        client_manager: ClientManager,
        noise_multiplier: int,
        batch_size: int,
        num_server_rounds: int,
        strategy: Scaffold,
        local_epochs: Optional[int] = None,
        local_steps: Optional[int] = None,
        delta: Optional[float] = None,
        wandb_reporter: Optional[ServerWandBReporter] = None,
        checkpointer: Optional[TorchCheckpointer] = None,
        warm_start: bool = False,
    ) -> None:
        ScaffoldServer.__init__(
            self,
            client_manager=client_manager,
            strategy=strategy,
            wandb_reporter=wandb_reporter,
            checkpointer=checkpointer,
            warm_start=warm_start,
        )
        InstanceLevelDPServer.__init__(
            self,
            client_manager=client_manager,
            strategy=strategy,
            noise_multiplier=noise_multiplier,
            local_epochs=local_epochs,
            local_steps=local_steps,
            batch_size=batch_size,
            delta=delta,
            num_server_rounds=num_server_rounds,
        )

    def fit(self, num_rounds: int, timeout: Optional[float]) -> History:
        """
        Run DP Scaffold algorithm for the specified number of rounds.
        """
        assert isinstance(self.strategy, Scaffold)

        # Initialize parameters attribute (specifc to the Scaffold algo)
        self.initialize_paramameters(timeout=timeout)

        # Now that we initialized the parameters for scaffold, call instance level privacy fit
        return InstanceLevelDPServer.fit(self, num_rounds=num_rounds, timeout=timeout)
