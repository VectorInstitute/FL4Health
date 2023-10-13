import os
from logging import INFO
from typing import Any, Dict, List, Optional, Tuple

import wandb
from flwr.common.logger import log
from flwr.common.typing import Scalar
from flwr.server.history import History
from wandb.wandb_run import Run


class WandBReporter:
    def __init__(
        self,
        project_name: str,
        run_name: str,
        group_name: str,
        entity: str,
        notes: Optional[str],
        tags: Optional[List],
        config: Dict[str, Any],
        local_log_directory: str = "./fl_wandb_logs",
    ) -> None:
        # Name of the project underwhich to store all of the logged values
        self.project_name = project_name
        # Name of the run under the group (server or client associated)
        self.run_name = run_name
        # Name of grouping on the W and B dashboard
        self.group_name = group_name
        # W and B username under which these experiments are to be logged
        self.entity = entity
        # Any notes to go along with the experiment to be logged
        self.notes = notes
        # Any tags to make searching easier.\
        self.tags = tags
        # Initialize the WandB logger
        self._maybe_create_local_log_directory(local_log_directory)
        wandb.init(
            dir=local_log_directory,
            project=self.project_name,
            name=self.run_name,
            group=self.group_name,
            entity=self.entity,
            notes=self.notes,
            tags=self.tags,
            config=config,
        )
        assert wandb.run is not None
        self.wandb_run: Run = wandb.run

    def _maybe_create_local_log_directory(self, local_log_directory: str) -> None:
        log_directory_exists = os.path.isdir(local_log_directory)
        if not log_directory_exists:
            os.mkdir(local_log_directory)
            log(INFO, f"Logging directory {local_log_directory} does not exist. Creating it.")
        else:
            log(INFO, f"Logging directory {local_log_directory} exists.")

    def _log_metrics(self, metric_dict: Dict[str, Any]) -> None:
        self.wandb_run.log(metric_dict)

    def shutdown_reporter(self) -> None:
        self.wandb_run.finish()


class ServerWandBReporter(WandBReporter):
    def __init__(
        self,
        project_name: str,
        run_name: str,
        group_name: str,
        entity: str,
        notes: Optional[str],
        tags: Optional[List],
        fl_config: Dict[str, Any],
    ) -> None:
        super().__init__(project_name, run_name, group_name, entity, notes, tags, fl_config)

    def _convert_losses_history(
        self, history_to_log: List[Dict[str, Scalar]], loss_history: List[Tuple[int, float]], loss_stub: str
    ) -> None:
        for server_round, loss in loss_history:
            # Server rounds are indexed starting at 1
            history_to_log[server_round - 1][f"{loss_stub}_loss"] = loss

    def _flatten_metrics_history(
        self,
        history_to_log: List[Dict[str, Scalar]],
        metrics_history: Dict[str, List[Tuple[int, Scalar]]],
        metric_stub: str,
    ) -> None:
        for metric_name, metric_history in metrics_history.items():
            metric = f"{metric_stub}_{metric_name}"
            for server_round, history_value in metric_history:
                # Server rounds are indexed starting at 1
                history_to_log[server_round - 1][metric] = history_value

    def report_metrics(self, server_rounds: int, history: History) -> None:
        # Servers construct a history object that collects aggregated metrics over the set of server rounds conducted.
        # So we need to reformat the history object into a W and B loggable object
        history_to_log: List[Dict[str, Scalar]] = [
            {"server_round": server_round} for server_round in range(server_rounds)
        ]
        if len(history.losses_centralized) > 0:
            self._convert_losses_history(history_to_log, history.losses_centralized, "centralized")
        if len(history.losses_distributed) > 0:
            self._convert_losses_history(history_to_log, history.losses_distributed, "distributed_val")
        if history.metrics_centralized:
            self._flatten_metrics_history(history_to_log, history.metrics_centralized, "centralized")
        if history.metrics_distributed:
            self._flatten_metrics_history(history_to_log, history.metrics_distributed, "distributed")
        if history.metrics_distributed_fit:
            self._flatten_metrics_history(history_to_log, history.metrics_distributed_fit, "distributed_fit")

        for server_metrics in history_to_log:
            self._log_metrics(server_metrics)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> Optional["ServerWandBReporter"]:
        assert "reporting_config" in config
        reporter_config = config["reporting_config"]
        if reporter_config["enabled"]:
            # Strip out the reporting configuration variables.
            fl_config = {key: value for key, value in config.items() if key != "reporting_config"}
            return ServerWandBReporter(
                reporter_config["project_name"],
                reporter_config["run_name"],
                reporter_config["group_name"],
                reporter_config["entity"],
                reporter_config.get("notes"),
                reporter_config.get("tags"),
                fl_config,
            )
        else:
            return None


class ClientWandBReporter(WandBReporter):
    def __init__(
        self,
        client_name: str,
        project_name: str,
        group_name: str,
        entity: str,
    ) -> None:
        self.client_name = client_name
        config = {"client_name": client_name}
        run_name = f"Client_{client_name}"
        super().__init__(project_name, run_name, group_name, entity, None, None, config)

    def log_model_type(self, model_type: str) -> None:
        self.add_client_model_type(f"{self.client_name}_model", model_type)

    def report_metrics(self, metrics: Dict[str, Any]) -> None:
        # Attach client name for W and B logging
        client_metrics = {f"{self.client_name}_{key}": metric for key, metric in metrics.items()}
        self._log_metrics(client_metrics)

    def add_client_model_type(self, client_name: str, model_type: str) -> None:
        self.wandb_run.config[client_name] = model_type

    @classmethod
    def from_config(cls, client_name: str, config: Dict[str, Any]) -> Optional["ClientWandBReporter"]:
        if "reporting_enabled" in config and config["reporting_enabled"]:
            return ClientWandBReporter(client_name, config["project_name"], config["group_name"], config["entity"])
        else:
            return None
