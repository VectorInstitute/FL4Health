from logging import INFO
import timeit
import torch.nn as nn
from typing import Optional
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.strategy import Strategy
from flwr.server.history import History

from fl4health.parameter_exchange.parameter_exchanger_base import ParameterExchanger
from fl4health.server.base_server import FlServerWithCheckpointing
from fl4health.checkpointing.checkpointer import TorchCheckpointer, ServerPerEpochCheckpointer
from fl4health.reporting.fl_wanb import ServerWandBReporter

from research.picai.fl_utils import get_initial_model_parameters


class PicaiServer(FlServerWithCheckpointing):
    def __init__(
        self,
        client_manager: ClientManager,
        model: nn.Module,
        parameter_exchanger: ParameterExchanger,
        strategy: Optional[Strategy] = None,
        wandb_reporter: Optional[ServerWandBReporter] = None,
        checkpointer: Optional[TorchCheckpointer] = None,
        per_epoch_checkpointer: ServerPerEpochCheckpointer = ServerPerEpochCheckpointer(
            checkpoint_dir="./", checkpoint_name="server_ckpt.pt"
        )
    ) -> None:
        assert wandb_reporter is None
        super().__init__(
            client_manager=client_manager,
            model=model,
            parameter_exchanger=parameter_exchanger,
            strategy=strategy,
            wandb_reporter=wandb_reporter,
            checkpointer=checkpointer
        )
        self.per_epoch_checkpointer = per_epoch_checkpointer

    def fit(self, num_rounds: int, timeout: Optional[float]) -> History:
        history = self.fit_with_per_epoch_checkpointing(num_rounds, timeout)
        if self.wandb_reporter:
            # report history to W and B
            self.wandb_reporter.report_metrics(num_rounds, history)
        return history

    def fit_with_per_epoch_checkpointing(self, num_rounds: int, timeout: Optional[float]) -> History:
        """Run federated averaging for a number of rounds."""

        # Initialize parameters
        log(INFO, "Initializing global parameters")

        if not self.per_epoch_checkpointer.checkpoint_exists():
            self.parameters = self._get_initial_parameters(timeout)
            self._hydrate_model_for_checkpointing()
            self.per_epoch_checkpointer.save_checkpoint({
                "model": self.server_model,
                "history": History(),
                "server_round": 1
            })

        model, history, server_round = self.per_epoch_checkpointer.load_checkpoint()
        self.parameters = get_initial_model_parameters(model)

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

            self._hydrate_model_for_checkpointing()
            self.per_epoch_checkpointer.save_checkpoint({
                "model": self.server_model,
                "history": history,
                "server_round": current_round
            })

        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        log(INFO, "FL finished in %s", elapsed)
        return history
