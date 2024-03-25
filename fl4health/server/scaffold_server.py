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
from fl4health.strategies.strategy_with_poll import StrategyWithPolling
import timeit
import os, json

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

class DPScaffoldLoggingServer(DPScaffoldServer):
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
        super().__init__(
            client_manager=client_manager,
            noise_multiplier=noise_multiplier,
            batch_size=batch_size,
            num_server_rounds=num_server_rounds,
            strategy=strategy,
            local_epochs=local_epochs,
            local_steps=local_steps,
            delta=delta,
            wandb_reporter=wandb_reporter,
            checkpointer=checkpointer,
            warm_start=warm_start
        )

        dir = checkpointer.best_checkpoint_path

        metrics_dir = os.path.join(os.path.dirname(dir), 
            'metrics'
        )

        if not os.path.exists(metrics_dir):
            os.makedirs(metrics_dir)

        self.metrics_path = os.path.join(
            metrics_dir,
            'server_metrics.json'
        )

        with open(self.metrics_path, 'w+') as file:
            json.dump({
                'privacy_hyperparameters': {
                    'noise_multiplier': noise_multiplier
                }
            }, file)

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

        assert isinstance(self.strategy, StrategyWithPolling)
        sample_counts = self.poll_clients_for_sample_counts(timeout)
        log(INFO, 'sample_counts')
        log(INFO, sample_counts)
        if len(sample_counts) > 1:
            self.setup_privacy_accountant(sample_counts)
        else:
            log(INFO, f'invalid sample count: {sample_counts}')

        history = History()

        # Initialize parameters
        log(INFO, "Initializing global parameters")
        self.parameters = self._get_initial_parameters(timeout=timeout)
        log(INFO, "Evaluating initial parameters")
        res = self.strategy.evaluate(0, parameters=self.parameters)
        if res is not None:
            log(
                INFO,
                "initial parameters (loss, other metrics): %s, %s",
                res[0],
                res[1],
            )
            history.add_loss_centralized(server_round=0, loss=res[0])
            history.add_metrics_centralized(server_round=0, metrics=res[1])

        # Run federated learning for num_rounds
        log(INFO, "FL starting")
        start_time = timeit.default_timer()

        for current_round in range(1, num_rounds + 1):
            # Train model and replace previous global model
            res_fit = self.fit_round(server_round=current_round, timeout=timeout)
            if res_fit:
                parameters_prime, fit_metrics, _ = res_fit  # fit_metrics_aggregated
                if parameters_prime:
                    self.parameters = parameters_prime
                history.add_metrics_distributed_fit(
                    server_round=current_round, metrics=fit_metrics
                )

            # Evaluate model using strategy implementation
            res_cen = self.strategy.evaluate(current_round, parameters=self.parameters)
            if res_cen is not None:
                loss_cen, metrics_cen = res_cen
                log(
                    INFO,
                    "fit progress: (%s, %s, %s, %s)",
                    current_round,
                    loss_cen,
                    metrics_cen,
                    timeit.default_timer() - start_time,
                )
                history.add_loss_centralized(server_round=current_round, loss=loss_cen)
                history.add_metrics_centralized(
                    server_round=current_round, metrics=metrics_cen
                )

            # Evaluate model on a sample of available clients
            res_fed = self.evaluate_round(server_round=current_round, timeout=timeout)
            if res_fed:
                loss_fed, evaluate_metrics_fed, _ = res_fed
                if loss_fed:
                    history.add_loss_distributed(
                        server_round=current_round, loss=loss_fed
                    )
                    history.add_metrics_distributed(
                        server_round=current_round, metrics=evaluate_metrics_fed
                    )

                    with open(self.metrics_path, 'r') as file:
                        metrics_to_save = json.load(file)

                    metrics_to_save['current_round'] = current_round

                    if current_round == 1:
                        metrics_to_save['privacy_hyperparameters']['num_fl_rounds'] = num_rounds


                    for key, value in fit_metrics.items():
                        if key not in metrics_to_save:
                            metrics_to_save[key] = [value]
                        else:
                            metrics_to_save[key].append(value)

                    if 'loss' not in metrics_to_save:
                        metrics_to_save['loss'] = [loss_fed]
                    else:
                        metrics_to_save['loss'].append(loss_fed)

                    for key, value in evaluate_metrics_fed.items():
                        if key not in metrics_to_save:
                            metrics_to_save[key] = [value]
                        else:
                            metrics_to_save[key].append(value)

                    # NOTE server only 
                    now = timeit.default_timer()
                    if 'time' not in metrics_to_save:
                        metrics_to_save['time'] = [now-start_time]
                    else:
                        metrics_to_save['time'].append(now-start_time)

                with open(self.metrics_path, 'w') as file:
                    json.dump(metrics_to_save, file)
                    log(DEBUG, f'finished recording metrics for round {current_round}')

        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        log(INFO, "FL finished in %s", elapsed)

        if self.wandb_reporter:
            # report history to W and B
            self.wandb_reporter.report_metrics(num_rounds, history)
        return history
