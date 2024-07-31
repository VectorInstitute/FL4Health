import timeit
from logging import INFO
from pathlib import Path
from typing import Optional, Tuple

import torch.nn as nn
from flwr.common.logger import log
from flwr.common.parameter import ndarrays_to_parameters
from flwr.common.typing import Parameters
from flwr.server.client_manager import ClientManager
from flwr.server.history import History
from flwr.server.strategy import Strategy

from fl4health.checkpointing.checkpointer import ServerPerRoundCheckpointer, TorchCheckpointer
from fl4health.parameter_exchange.parameter_exchanger_base import ParameterExchanger
from fl4health.reporting.fl_wandb import ServerWandBReporter
from fl4health.server.base_server import FlServerWithCheckpointing


def get_initial_model_parameters(client_model: nn.Module) -> Parameters:
    # Initializing the model parameters on the server side.
    return ndarrays_to_parameters([val.cpu().numpy() for _, val in client_model.state_dict().items()])


class PicaiServer(FlServerWithCheckpointing):
    def __init__(
        self,
        client_manager: ClientManager,
        model: nn.Module,
        parameter_exchanger: ParameterExchanger,
        strategy: Optional[Strategy] = None,
        wandb_reporter: Optional[ServerWandBReporter] = None,
        checkpointer: Optional[TorchCheckpointer] = None,
        intermediate_checkpoint_dir: Path = Path("./"),
    ) -> None:
        """
        A simple extension of the FlServerWithCheckpointing that adds tolerance to preemptions by checkpointing
        the server state each round and loading from checkpoint on initialization if it exists.

        Args:
            client_manager (ClientManager): Determines the mechanism by which clients are sampled by the server, if
                they are to be sampled at all.
            strategy (Optional[Strategy], optional): The aggregation strategy to be used by the server to handle.
                client updates and other information potentially sent by the participating clients. If None the
                strategy is FedAvg as set by the flwr Server.
            wandb_reporter (Optional[ServerWandBReporter], optional): To be provided if the server is to log
                information and results to a Weights and Biases account. If None is provided, no logging occurs.
                Defaults to None.
            checkpointer (Optional[TorchCheckpointer], optional): To be provided if the server should perform
                server side checkpointing based on some criteria. If none, then no server-side checkpointing is
                performed. Defaults to None.
            intermediate_checkpoint_dir (Path): A directory to store and load checkpoints from for the server
                during an FL experiment.
        """
        assert wandb_reporter is None
        super().__init__(
            client_manager=client_manager,
            model=model,
            parameter_exchanger=parameter_exchanger,
            strategy=strategy,
            wandb_reporter=wandb_reporter,
            checkpointer=checkpointer,
        )
        self.per_round_checkpointer = ServerPerRoundCheckpointer(intermediate_checkpoint_dir, Path("server.ckpt"))

    def fit(self, num_rounds: int, timeout: Optional[float]) -> Tuple[History, float]:
        """
        Overrides method in parent class to call custom fit_with_per_round_checkpointing that is resilient
        against pre-emptions.

        Args:
            num_rounds (int): The number of rounds to perform federated learning.
            timeout (Optional[float]): The timeout for clients to return results in a given FL round.

        Returns:
            Tuple[History, float]: The first element of the tuple is a history object containing the losses and
                metrics computed during training and validation. The second element of the tuple is
                the elapsed time in seconds.
        """
        history, elapsed_time = self.fit_with_per_epoch_checkpointing(num_rounds, timeout)
        if self.wandb_reporter:
            # report history to W and B
            self.wandb_reporter.report_metrics(num_rounds, history)
        return history, elapsed_time

    def fit_with_per_epoch_checkpointing(self, num_rounds: int, timeout: Optional[float]) -> Tuple[History, float]:
        """
        Runs federated learning for a number of rounds. Heavily based on the fit method from the base
        server provided by flower (flwr.server.server.Server) except that it is resilient to pre-emptions.
        It accomplishes this by checkpointing the sever state each round. In the case of pre-emption,
        when the server is restarted it will load from the most recent checkpoint.

        Args:
            num_rounds (int): The number of rounds to perform federated learning.
            timeout (Optional[float]): The timeout for clients to return results in a given FL round.

        Returns:
            Tuple[History, float]: The first element of the tuple is a history object containing the losses and
                metrics computed during training and validation. The second element of the tuple is
                the elapsed time in seconds.
        """
        # Initialize parameters
        log(INFO, "Initializing global parameters")

        # if checkpoint exists, update history, server round and model accordingly
        if self.per_round_checkpointer.checkpoint_exists():
            model, history, server_round = self.per_round_checkpointer.load_checkpoint()
            self.parameters = get_initial_model_parameters(model)
        else:
            self.parameters = self._get_initial_parameters(server_round=1, timeout=timeout)
            history = History()
            server_round = 1

        if server_round == 1:
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

        for current_round in range(server_round, num_rounds + 1):
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

            # Save checkpoint after training and testing
            self._hydrate_model_for_checkpointing()
            self.per_round_checkpointer.save_checkpoint(
                {"model": self.server_model, "history": history, "server_round": current_round}
            )

        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed_time = end_time - start_time
        log(INFO, "FL finished in %s", elapsed_time)
        return history, elapsed_time
