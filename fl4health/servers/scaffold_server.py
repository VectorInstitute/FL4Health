from collections.abc import Callable, Sequence
from logging import DEBUG, ERROR, INFO

from flwr.common import Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.logger import log
from flwr.common.typing import Config, Scalar
from flwr.server.client_manager import ClientManager
from flwr.server.history import History
from flwr.server.server import fit_clients

from fl4health.checkpointing.server_module import (
    DpScaffoldServerCheckpointAndStateModule,
    ScaffoldServerCheckpointAndStateModule,
)
from fl4health.reporting.base_reporter import BaseReporter
from fl4health.servers.base_server import FlServer
from fl4health.servers.instance_level_dp_server import InstanceLevelDpServer
from fl4health.strategies.scaffold import OpacusScaffold, Scaffold


class ScaffoldServer(FlServer):
    def __init__(
        self,
        client_manager: ClientManager,
        fl_config: Config,
        strategy: Scaffold,
        reporters: Sequence[BaseReporter] | None = None,
        checkpoint_and_state_module: ScaffoldServerCheckpointAndStateModule | None = None,
        on_init_parameters_config_fn: Callable[[int], dict[str, Scalar]] | None = None,
        server_name: str | None = None,
        accept_failures: bool = True,
        warm_start: bool = False,
    ) -> None:
        """
        Custom FL Server for scaffold algorithm to handle warm initialization of control variates as specified in
        https://arxiv.org/abs/1910.06378.

        Args:
            client_manager (ClientManager): Determines the mechanism by which clients are sampled by the server, if
                they are to be sampled at all.
            fl_config (Config): This should be the configuration that was used to setup the federated training.
                In most cases it should be the "source of truth" for how FL training/evaluation should proceed. For
                example, the config used to produce the ``on_fit_config_fn`` and ``on_evaluate_config_fn`` for the
                strategy.

                **NOTE**: This config is **DISTINCT** from the Flwr server config, which is extremely minimal.
            strategy (Scaffold): The aggregation strategy to be used by the server to handle client updates and
                other information potentially sent by the participating clients. This strategy must be of SCAFFOLD
                type.
            reporters (Sequence[BaseReporter] | None, optional): sequence of FL4Health reporters which the server
                should send data to before and after each round. Defaults to None.
            checkpoint_and_state_module (ScaffoldServerCheckpointAndStateModule | None, optional): This module is used
                to handle both model checkpointing and state checkpointing. The former is aimed at saving model
                artifacts to be used or evaluated after training. The latter is used to preserve training state
                (including models) such that if FL training is interrupted, the process may be restarted. If no
                module is provided, no checkpointing or state preservation will happen. Defaults to None.
            on_init_parameters_config_fn (Callable[[int], dict[str, Scalar]] | None, optional): Function used to
                configure how one asks a client to provide parameters from which to initialize all other clients by
                providing a ``Config`` dictionary. If this is none, then a blank config is sent with the parameter
                request (which is default behavior for flower servers). Defaults to None.
            server_name (str | None, optional): An optional string name to uniquely identify server. This name is also
                used as part of any state checkpointing done by the server. Defaults to None.
            accept_failures (bool, optional): Determines whether the server should accept failures during training or
                evaluation from clients or not. If set to False, this will cause the server to shutdown all clients
                and throw an exception. Defaults to True.
            warm_start (bool, optional):  Whether or not to initialize control variates of each client as local
                gradients. The clients will perform a training pass (without updating the weights) in order to provide
                a "warm" estimate of the SCAFFOLD control variates. If false, variates are initialized to 0.
                Defaults to False.
        """
        if checkpoint_and_state_module is not None:
            assert isinstance(
                checkpoint_and_state_module,
                ScaffoldServerCheckpointAndStateModule,
            ), "checkpoint_and_state_module must have type ScaffoldServerCheckpointAndStateModule"
        FlServer.__init__(
            self,
            client_manager=client_manager,
            fl_config=fl_config,
            strategy=strategy,
            reporters=reporters,
            checkpoint_and_state_module=checkpoint_and_state_module,
            on_init_parameters_config_fn=on_init_parameters_config_fn,
            server_name=server_name,
            accept_failures=accept_failures,
        )
        self.warm_start = warm_start

    def _get_initial_parameters(self, server_round: int, timeout: float | None) -> Parameters:
        """
        Overrides the ``_get_initial_parameters`` in the flwr server base class to strap on the possibility of a
        ``warm_start`` for SCAFFOLD. Initializes parameters (models weights and control variates) of the server.
        If warm_start is True, control variates are initialized as the the average local client-side gradients
        (while model weights remain unchanged). That is, all of the clients run a training pass, but the trained
        weights are discarded.

        Args:
            server_round (int): The current server round.
            timeout (float | None): If the server strategy object does not have a server-side initial parameters
                function defined, then one of the clients is polled and their model parameters are returned in order to
                initialize the models of all clients. Timeout defines how long to wait for a response.

        Returns:
            (Parameters): Initial parameters (model weights and control variates).
        """
        assert isinstance(self.strategy, Scaffold)
        # First run basic parameter initialization from the parent server
        initial_parameters = super()._get_initial_parameters(server_round, timeout=timeout)

        # If warm_start, run routine to initialize control variates without updating global model
        # control variates are initialized as average local gradient over training steps
        # while the model weights remain unchanged (until the FL rounds start)
        if self.warm_start:
            log(
                INFO,
                "Using Warm Start Strategy. Waiting for clients to be available for polling",
            )
            client_instructions = self.strategy.configure_fit_all(
                server_round=0,
                parameters=initial_parameters,
                client_manager=self._client_manager,
            )
            if not client_instructions:
                log(ERROR, "Warm Start initialization failed: no clients selected", 1)
            else:
                log(
                    DEBUG,
                    f"Warm start: strategy sampled {len(client_instructions)} \
                    clients (out of {self._client_manager.num_available()})",
                )

                results, failures = fit_clients(
                    client_instructions,
                    self.max_workers,
                    timeout,
                    group_id=server_round,
                )

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

    def fit(self, num_rounds: int, timeout: float | None) -> tuple[History, float]:
        """
        Run the SCAFFOLD FL algorithm for a fixed number of rounds. This overrides the base server fit class just to
        ensure that the provided strategy is a Scaffold strategy object before proceeding.

        Args:
            num_rounds (int): Number of rounds of FL to perform (i.e. server rounds).
            timeout (float | None): Timeout associated with queries to the clients in seconds. The server waits for
                timeout seconds before moving on without any unresponsive clients. If None, there is no timeout and the
                server waits for the minimum number of clients to be available set in the strategy.

        Returns:
            (tuple[History, float]): The first element of the tuple is a ``History`` object containing the full set of
                FL training results, including things like aggregated loss and metrics. Tuple also includes elapsed
                time in seconds for round.
        """
        assert isinstance(self.strategy, Scaffold)
        return super().fit(num_rounds=num_rounds, timeout=timeout)


class DPScaffoldServer(ScaffoldServer, InstanceLevelDpServer):
    def __init__(
        self,
        client_manager: ClientManager,
        fl_config: Config,
        noise_multiplier: int,
        batch_size: int,
        num_server_rounds: int,
        strategy: OpacusScaffold,
        local_epochs: int | None = None,
        local_steps: int | None = None,
        delta: float | None = None,
        checkpoint_and_state_module: DpScaffoldServerCheckpointAndStateModule | None = None,
        warm_start: bool = False,
        reporters: Sequence[BaseReporter] | None = None,
        on_init_parameters_config_fn: Callable[[int], dict[str, Scalar]] | None = None,
        server_name: str | None = None,
        accept_failures: bool = True,
    ) -> None:
        """
        Custom FL Server for Instance Level Differentially Private Scaffold algorithm as specified in
        https://arxiv.org/abs/2111.09278.

        Args:
            client_manager (ClientManager): Determines the mechanism by which clients are sampled by the server, if
                they are to be sampled at all.
            fl_config (Config): This should be the configuration that was used to setup the federated training.
                In most cases it should be the "source of truth" for how FL training/evaluation should proceed. For
                example, the config used to produce the ``on_fit_config_fn`` and ``on_evaluate_config_fn`` for the
                strategy.

                **NOTE**: This config is **DISTINCT** from the Flwr server config, which is extremely minimal.
            noise_multiplier (int): The amount of Gaussian noise to be added to the per sample gradient during
                DP-SGD.
            batch_size (int): The batch size to be used in training on the client-side. Used in privacy accounting.
            num_server_rounds (int): The number of server rounds to be done in FL training. Used in privacy accounting
            local_epochs (int | None, optional): Number of local epochs to be performed on the client-side. This is
                used in privacy accounting. One of ``local_epochs`` or ``local_steps`` should be defined, but not both.
                Defaults to None.
            local_steps (int | None, optional): Number of local steps to be performed on the client-side. This is
                used in privacy accounting. One of ``local_epochs`` or ``local_steps`` should be defined, but not both.
                Defaults to None.
            strategy (Scaffold): The aggregation strategy to be used by the server to handle client updates and
                other information potentially sent by the participating clients. This strategy must be of SCAFFOLD
                type.
            checkpoint_and_state_module (DpScaffoldServerCheckpointAndStateModule | None, optional): This module is
                used to handle both model checkpointing and state checkpointing. The former is aimed at saving model
                artifacts to be used or evaluated after training. The latter is used to preserve training state
                (including models) such that if FL training is interrupted, the process may be restarted. If no
                module is provided, no checkpointing or state preservation will happen. Defaults to None.
            warm_start (bool, optional): Whether or not to initialize control variates of each client as local
                gradients. The clients will perform a training pass (without updating the weights) in order to provide
                a "warm" estimate of the SCAFFOLD control variates. If false, variates are initialized to 0.
                Defaults to False.
            delta (float | None, optional): The delta value for epsilon-delta DP accounting. If None it defaults to
                being ``1/total_samples`` in the FL run. Defaults to None.
            reporters (Sequence[BaseReporter], optional): A sequence of FL4Health reporters which the client should
                send data to.
            on_init_parameters_config_fn (Callable[[int], dict[str, Scalar]] | None, optional): Function used to
                configure how one asks a client to provide parameters from which to initialize all other clients by
                providing a ``Config`` dictionary. If this is none, then a blank config is sent with the parameter
                request (which is default behavior for flower servers). Defaults to None.
            server_name (str | None, optional): An optional string name to uniquely identify server. This name is also
                used as part of any state checkpointing done by the server. Defaults to None.
            accept_failures (bool, optional): Determines whether the server should accept failures during training or
                evaluation from clients or not. If set to False, this will cause the server to shutdown all clients
                and throw an exception. Defaults to True.
        """
        # Require the strategy to be an  OpacusStrategy to handle the Opacus model conversion etc.
        assert isinstance(strategy, OpacusScaffold), (
            f"Strategy much be of type OpacusScaffold to handle Opacus models but is of type {type(strategy)}"
        )
        ScaffoldServer.__init__(
            self,
            client_manager=client_manager,
            fl_config=fl_config,
            strategy=strategy,
            warm_start=warm_start,
            reporters=reporters,
            on_init_parameters_config_fn=on_init_parameters_config_fn,
            checkpoint_and_state_module=checkpoint_and_state_module,
            server_name=server_name,
            accept_failures=accept_failures,
        )
        InstanceLevelDpServer.__init__(
            self,
            client_manager=client_manager,
            fl_config=fl_config,
            noise_multiplier=noise_multiplier,
            num_server_rounds=num_server_rounds,
            batch_size=batch_size,
            strategy=strategy,
            local_epochs=local_epochs,
            local_steps=local_steps,
            delta=delta,
            on_init_parameters_config_fn=on_init_parameters_config_fn,
            server_name=server_name,
            accept_failures=accept_failures,
        )

    def fit(self, num_rounds: int, timeout: float | None) -> tuple[History, float]:
        """
        Run DP Scaffold algorithm for the specified number of rounds.

        Args:
            num_rounds (int): Number of rounds of FL to perform (i.e. server rounds).
            timeout (float | None): Timeout associated with queries to the clients in seconds. The server waits for
                timeout seconds before moving on without any unresponsive clients. If None, there is no timeout and the
                server waits for the minimum number of clients to be available set in the strategy.

        Returns:
            (tuple[History, float]): First element of tuple is history object containing the full set of FL training
                results, including aggregated loss and metrics. Tuple also includes the elapsed time in seconds for
                round.
        """
        assert isinstance(self.strategy, Scaffold)
        # Now that we initialized the parameters for scaffold, call instance level privacy fit
        return InstanceLevelDpServer.fit(self, num_rounds=num_rounds, timeout=timeout)
