from logging import DEBUG, ERROR, INFO
from typing import Optional

from flwr.common import Parameters, ndarrays_to_parameters, parameters_to_ndarrays
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
    def __init__(
        self,
        client_manager: ClientManager,
        strategy: Scaffold,
        wandb_reporter: Optional[ServerWandBReporter] = None,
        checkpointer: Optional[TorchCheckpointer] = None,
        warm_start: bool = False,  # Whether or not to initialize control variates of each client as local gradient
    ) -> None:
        """
        Custom FL Server for scaffold algorithm to handle warm initialization of control variates
        as specified in https://arxiv.org/abs/1910.06378.

        Args:
            client_manager (ClientManager): Determines the mechanism by which clients are sampled by the server, if
                they are to be sampled at all.
            strategy (Scaffold): The aggregation strategy to be used by the server to handle client updates and
                other information potentially sent by the participating clients. This strategy must be of SCAFFOLD
                type.
            wandb_reporter (Optional[ServerWandBReporter], optional): To be provided if the server is to log
                information and results to a Weights and Biases account. If None is provided, no logging occurs.
                Defaults to None.
            checkpointer (Optional[TorchCheckpointer], optional): To be provided if the server should perform
                server side checkpointing based on some criteria. If none, then no server-side checkpointing is
                performed. Defaults to None.
            warm_start (bool, optional): Whether or not to initialize control variates of each client as
            local gradients. The clients will perform a training pass (without updating the weights) in order to
                provide a "warm" estimate of the SCAFFOLD control variates. If false, variates are initialized to 0.
                Defaults to False.
        """
        FlServer.__init__(
            self,
            client_manager=client_manager,
            strategy=strategy,
            wandb_reporter=wandb_reporter,
            checkpointer=checkpointer,
        )
        self.warm_start = warm_start

    def _get_initial_parameters(self, timeout: Optional[float]) -> Parameters:
        """
        Overrides the _get_initial_parameters in the flwr server base class to strap on the possibility of a
        warm_start for SCAFFOLD. Initializes parameters (models weights and control variates) of the server.
        If warm_start is True, control variates are initialized as the the average local client-side gradients
        (while model weights remain unchanged). That is, all of the clients run a training pass, but the trained
        weights are discarded.

        Args:
            timeout (Optional[float]): If the server strategy object does not have a server-side initial parameters
                function defined, then one of the clients is polled and their model parameters are returned in order to
                initialize the models of all clients. Timeout defines how long to wait for a response.
        """
        assert isinstance(self.strategy, Scaffold)
        # First run basic parameter initialization from the parent server
        initial_parameters = super()._get_initial_parameters(timeout=timeout)

        # If warm_start, run routine to initialize control variates without updating global model
        # control variates are initialized as average local gradient over training steps
        # while the model weights remain unchanged (until the FL rounds start)
        if self.warm_start:
            log(INFO, "Using Warm Start Strategy. Waiting for clients to be available for polling")
            client_instructions = self.strategy.configure_fit_all(
                server_round=0, parameters=initial_parameters, client_manager=self._client_manager
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
                    parameters_to_ndarrays(initial_parameters)
                )

                # Get new parameters by combining original weights with server control variates from warm start
                initial_parameters = ndarrays_to_parameters(
                    self.strategy.parameter_packer.pack_parameters(initial_weights, server_control_variates)
                )

        return initial_parameters

    def fit(self, num_rounds: int, timeout: Optional[float]) -> History:
        """
        Run the SCAFFOLD FL algortihm for a fixed number of rounds. This overrides the base server fit class just to
        ensure that the provided strategy is a Scaffold strategy object before proceeding.

        Args:
            num_rounds (int): Number of rounds of FL to perform (i.e. server rounds).
            timeout (Optional[float]): Timeout associated with queries to the clients in seconds. The server waits for
                timeout seconds before moving on without any unresponsive clients. If None, there is no timeout and the
                server waits for the minimum number of clients to be available set in the strategy.

        Returns:
            History: The history object contains the full set of FL training results, including things like aggregated
                loss and metrics.
        """
        assert isinstance(self.strategy, Scaffold)
        return super().fit(num_rounds=num_rounds, timeout=timeout)


class DPScaffoldServer(ScaffoldServer, InstanceLevelDPServer):
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
        """
        Custom FL Server for Instance Level Differentially Private Scaffold algorithm as specified in
        https://arxiv.org/abs/2111.09278

        Args:
            client_manager (ClientManager): Determines the mechanism by which clients are sampled by the server, if
                they are to be sampled at all.
            noise_multiplier (int): The amount of Gaussian noise to be added to the per sample gradient during
                DP-SGD.
            batch_size (int): The batch size to be used in training on the client-side. Used in privacy accounting.
            num_server_rounds (int): The number of server rounds to be done in FL training. Used in privacy accounting
            local_epochs (Optional[int], optional): Number of local epochs to be performed on the client-side. This is
                used in privacy accounting. One of local_epochs or local_steps should be defined, but not both.
                Defaults to None.
            local_steps (Optional[int], optional): Number of local steps to be performed on the client-side. This is
                used in privacy accounting. One of local_epochs or local_steps should be defined, but not both.
                Defaults to None.
            strategy (Scaffold): The aggregation strategy to be used by the server to handle client updates and
                other information potentially sent by the participating clients. This strategy must be of SCAFFOLD
                type.
            wandb_reporter (Optional[ServerWandBReporter], optional): To be provided if the server is to log
                information and results to a Weights and Biases account. If None is provided, no logging occurs.
                Defaults to None.
            checkpointer (Optional[TorchCheckpointer], optional): To be provided if the server should perform
                server side checkpointing based on some criteria. If none, then no server-side checkpointing is
                performed. Defaults to None.
            warm_start (bool, optional): Whether or not to initialize control variates of each client as
                local gradients. The clients will perform a training pass (without updating the weights) in order to
                provide a "warm" estimate of the SCAFFOLD control variates. If false, variates are initialized to 0.
                Defaults to False.
            delta (Optional[float], optional): The delta value for epislon-delta DP accounting. If None it defaults to
                being 1/total_samples in the FL run. Defaults to None.
        """
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

        Args:
            num_rounds (int): Number of rounds of FL to perform (i.e. server rounds).
            timeout (Optional[float]): Timeout associated with queries to the clients in seconds. The server waits for
                timeout seconds before moving on without any unresponsive clients. If None, there is no timeout and the
                server waits for the minimum number of clients to be available set in the strategy.

        Returns:
            History: The history object contains the full set of FL training results, including things like aggregated
                loss and metrics.
        """
        assert isinstance(self.strategy, Scaffold)
        # Now that we initialized the parameters for scaffold, call instance level privacy fit
        return InstanceLevelDPServer.fit(self, num_rounds=num_rounds, timeout=timeout)
