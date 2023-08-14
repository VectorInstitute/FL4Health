import timeit
from logging import DEBUG, ERROR, INFO
from typing import Optional

from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.history import History
from flwr.server.server import fit_clients

from fl4health.checkpointing.checkpointer import TorchCheckpointer
from fl4health.reporting.fl_wanb import ServerWandBReporter
from fl4health.server.base_server import FlServer
from fl4health.strategies.scaffold import Scaffold


class ScaffoldServer(FlServer):
    """
    Custom FL Server for scaffold algorithm to handle warm initialization of control variates.
    """

    def __init__(
        self,
        client_manager: ClientManager,
        strategy: Optional[Scaffold] = None,
        wandb_reporter: Optional[ServerWandBReporter] = None,
        checkpointer: Optional[TorchCheckpointer] = None,
        warm_start: bool = False,  # Whether or not to initialize control variates of each client as local gradient
    ) -> None:
        super().__init__(
            client_manager=client_manager, strategy=strategy, wandb_reporter=wandb_reporter, checkpointer=checkpointer
        )
        self.warm_start = warm_start

    def fit(self, num_rounds: int, timeout: Optional[float]) -> History:
        """Run Scaffold algorithm for a number of rounds."""

        assert isinstance(self.strategy, Scaffold)

        history = History()
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

                # Get new parameters by combining original weights with server control variates from warm start`
                self.parameters = ndarrays_to_parameters(
                    self.strategy.parameter_packer.pack_parameters(initial_weights, server_control_variates)
                )

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
                history.add_metrics_distributed_fit(server_round=current_round, metrics=fit_metrics)

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
                history.add_metrics_centralized(server_round=current_round, metrics=metrics_cen)

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
        return history
