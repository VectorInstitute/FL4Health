import datetime
from logging import DEBUG, ERROR, INFO, WARNING
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, TypeVar, Union

import torch.nn as nn
from flwr.common import EvaluateRes, Parameters
from flwr.common.logger import log
from flwr.common.parameter import parameters_to_ndarrays
from flwr.common.typing import Code, Config, GetParametersIns, Scalar
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.history import History
from flwr.server.server import EvaluateResultsAndFailures, FitResultsAndFailures, Server, evaluate_clients
from flwr.server.strategy import Strategy

from fl4health.checkpointing.checkpointer import PerRoundCheckpointer, TorchCheckpointer
from fl4health.parameter_exchange.parameter_exchanger_base import ParameterExchanger
from fl4health.reporting.base_reporter import BaseReporter
from fl4health.reporting.reports_manager import ReportsManager
from fl4health.servers.polling import poll_clients
from fl4health.strategies.strategy_with_poll import StrategyWithPolling
from fl4health.utils.config import narrow_dict_type_and_set_attribute
from fl4health.utils.metrics import TEST_LOSS_KEY, TEST_NUM_EXAMPLES_KEY, MetricPrefix
from fl4health.utils.parameter_extraction import get_all_model_parameters
from fl4health.utils.random import generate_hash
from fl4health.utils.typing import EvaluateFailures, FitFailures

ExchangerType = TypeVar("ExchangerType", bound=ParameterExchanger)


class FlServer(Server):
    def __init__(
        self,
        client_manager: ClientManager,
        fl_config: Config,
        strategy: Optional[Strategy] = None,
        reporters: Sequence[BaseReporter] | None = None,
        model: nn.Module | None = None,
        checkpointer: Union[TorchCheckpointer, Sequence[TorchCheckpointer]] | None = None,
        parameter_exchanger: ExchangerType | None = None,
        intermediate_server_state_dir: Path | None = None,
        on_init_parameters_config_fn: Callable[[int], Dict[str, Scalar]] | None = None,
        server_name: str | None = None,
        accept_failures: bool = True,
    ) -> None:
        """
        Base Server for the library to facilitate strapping additional/useful machinery to the base flwr server.

        Args:
            client_manager (ClientManager): Determines the mechanism by which clients are sampled by the server, if
                they are to be sampled at all.
            fl_config (Config): This should be the configuration that was used to setup the federated training.
                In most cases it should be the "source of truth" for how FL training/evaluation should proceed. For
                example, the config used to produce the on_fit_config_fn and on_evaluate_config_fn for the strategy.
                NOTE: This config is DISTINCT from the Flwr server config, which is extremely minimal.
            strategy (Optional[Strategy], optional): The aggregation strategy to be used by the server to handle.
                client updates and other information potentially sent by the participating clients. If None the
                strategy is FedAvg as set by the flwr Server. Defaults to None.
            reporters (Sequence[BaseReporter] | None, optional):  sequence of FL4Health reporters which the server
                should send data to before and after each round. Defaults to None.
            model (Optional[nn.Module]): This is the torch model to be checkpointed. It will be hydrated by the
                _hydrate_model_for_checkpointing function so that it has the proper weights to be saved. If no model
                is defined and checkpointing is attempted an error will throw. Defaults to None.
            checkpointer (Optional[Union[TorchCheckpointer, Sequence[TorchCheckpointer]]], optional): To be provided
                if the server should perform server side checkpointing based on some criteria. If none, then no
                server-side checkpointing is performed. Multiple checkpointers can also be passed in a sequence to
                checkpointer based on multiple criterion. Ensure checkpoint names are different for each checkpoint or
                they will overwrite on another. Defaults to None.
            parameter_exchanger (Optional[ExchangerType], optional): A parameter exchanger used to facilitate
                server-side model checkpointing if a checkpointer has been defined. If not provided then checkpointing
                will not be done unless the _hydrate_model_for_checkpointing function is overridden. Because the
                server only sees numpy arrays, the parameter exchanger is used to insert the numpy arrays into a
                provided model. Defaults to None.
            intermediate_server_state_dir (Path): A directory to store and load state from for the server
                during an FL experiment. This allows for the saving of server state in case federated training is
                interrupted and needs to be restarted from the same point. If none, then no state is saved during each
                server round. Defaults to None.
            on_init_parameters_config_fn (Optional[Callable[[int], Dict[str, Scalar]]], optional):
                Function used to configure how one asks a client to provide parameters from which to initialize all
                other clients by providing a Config dictionary. If this is none, then a blank config is sent with the
                parameter request (which is default behavior for flower servers). Defaults to None.
            server_name (Optional[str], optional): An optional string name to uniquely identify server.
                Defaults to None.
            accept_failures (bool, optional): Determines whether the server should accept failures during training or
                evaluation from clients or not. If set to False, this will cause the server to shutdown all clients
                and throw an exception. Defaults to True.
        """

        super().__init__(client_manager=client_manager, strategy=strategy)
        self.fl_config = fl_config
        self.server_model = model
        self.on_init_parameters_config_fn = on_init_parameters_config_fn
        self.checkpointer = [checkpointer] if isinstance(checkpointer, TorchCheckpointer) else checkpointer
        # To facilitate model rehydration from server-side state for checkpointing
        self.parameter_exchanger = parameter_exchanger
        self.server_name = server_name if server_name is not None else generate_hash()
        self.accept_failures = accept_failures

        self.per_round_checkpointer: PerRoundCheckpointer | None

        if intermediate_server_state_dir is not None:
            log(
                WARNING,
                "intermediate_server_state_dir is not None. Creating PerRoundCheckpointer. This functionality is "
                "still experimental and only supported for BasicClient and NnunetClient currently.",
            )
            self.per_round_checkpointer = PerRoundCheckpointer(
                intermediate_server_state_dir, Path(f"{self.server_name}.ckpt")
            )
        else:
            self.per_round_checkpointer = None

        self.current_round: int
        self.history: History

        # Initialize reporters with server name information.
        self.reports_manager = ReportsManager(reporters)
        self.reports_manager.initialize(id=self.server_name)
        self._log_fl_config()

    def update_before_fit(self, num_rounds: int, timeout: Optional[float]) -> None:
        """
        Hook method to allow the server to do some work before starting the fit process. In the base server, it is a
        no-op function, but it can be overridden in child classes for custom functionality. For example, the
        NnUNetServer class uses this method to ask a client to initialize the global nnunet plans if one is not
        provided in the config. This can only be done after the clients have started up and are ready to train.

        Args:
            num_rounds (int): The number of server rounds of FL to be performed
            timeout (Optional[float], optional): The server's timeout parameter. Useful if one is requesting
                information from a client. Defaults to None, which indicates indefinite timeout.
        """
        pass

    def report_centralized_eval(self, history: History, num_rounds: int) -> None:
        if len(history.losses_centralized) == 0:
            return

        # Parse and report history for loss and metrics on centralized validation set.
        for round in range(num_rounds):
            self.reports_manager.report(
                {"val - loss - centralized": history.losses_centralized[round][1]},
                round + 1,
            )
            round_metrics = {}
            for metric, vals in history.metrics_centralized.items():
                round_metrics.update({metric: vals[round][1]})
            self.reports_manager.report({"eval_round_metrics_centralized": round_metrics}, round + 1)

    def fit_with_per_round_checkpointing(self, num_rounds: int, timeout: Optional[float]) -> Tuple[History, float]:
        """
        Runs federated learning for a number of rounds. Heavily based on the fit method from the base
        server provided by flower (flwr.server.server.Server) except that it is resilient to preemptions.
        It accomplishes this by checkpointing the server state each round. In the case of preemption,
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

        assert self.per_round_checkpointer is not None

        # if checkpoint exists, update history, server round and model accordingly
        if self.per_round_checkpointer.checkpoint_exists():
            self._load_server_state()
        else:
            log(INFO, "Initializing server state")
            self.parameters = self._get_initial_parameters(server_round=0, timeout=timeout)
            self.history = History()
            self.current_round = 1

        if self.current_round == 1:
            log(INFO, "Evaluating initial parameters")
            res = self.strategy.evaluate(0, parameters=self.parameters)
            if res is not None:
                log(
                    INFO,
                    "initial parameters (loss, other metrics): %s, %s",
                    res[0],
                    res[1],
                )
                self.history.add_loss_centralized(server_round=0, loss=res[0])
                self.history.add_metrics_centralized(server_round=0, metrics=res[1])

            # Run federated learning for num_rounds
            log(INFO, "FL starting")

        start_time = datetime.datetime.now()

        while self.current_round < (num_rounds + 1):
            # Train model and replace previous global model
            res_fit = self.fit_round(server_round=self.current_round, timeout=timeout)
            if res_fit:
                parameters_prime, fit_metrics, _ = res_fit  # fit_metrics_aggregated
                if parameters_prime:
                    self.parameters = parameters_prime
                self.history.add_metrics_distributed_fit(server_round=self.current_round, metrics=fit_metrics)

            # Evaluate model using strategy implementation
            res_cen = self.strategy.evaluate(self.current_round, parameters=self.parameters)
            if res_cen is not None:
                loss_cen, metrics_cen = res_cen
                log(
                    INFO,
                    "fit progress: (%s, %s, %s, %s)",
                    self.current_round,
                    loss_cen,
                    metrics_cen,
                    (datetime.datetime.now() - start_time).total_seconds(),
                )
                self.history.add_loss_centralized(server_round=self.current_round, loss=loss_cen)
                self.history.add_metrics_centralized(server_round=self.current_round, metrics=metrics_cen)

            # Evaluate model on a sample of available clients
            res_fed = self.evaluate_round(server_round=self.current_round, timeout=timeout)
            if res_fed:
                loss_fed, evaluate_metrics_fed, _ = res_fed
                if loss_fed:
                    self.history.add_loss_distributed(server_round=self.current_round, loss=loss_fed)
                    self.history.add_metrics_distributed(server_round=self.current_round, metrics=evaluate_metrics_fed)

            self.current_round += 1

            # Save checkpoint after training and testing
            self._hydrate_model_for_checkpointing()
            self._save_server_state()

        # Bookkeeping
        end_time = datetime.datetime.now()
        elapsed_time = end_time - start_time
        log(INFO, "FL finished in %s", str(elapsed_time))
        return self.history, elapsed_time.total_seconds()

    def fit(self, num_rounds: int, timeout: Optional[float]) -> Tuple[History, float]:
        """
        Run federated learning for a number of rounds. This function also allows the server to perform some operations
        prior to fitting starting. This is useful, for example, if you need to communicate with the clients to
        initialize anything prior to FL starting (see nnunet server for an example)

        Args:
            num_rounds (int): Number of server rounds to run.
            timeout (Optional[float]): The amount of time in seconds that the server will wait for results from the
                clients selected to participate in federated training.

        Returns:
            Tuple[History, float]: The first element of the tuple is a history object containing the full set of
                FL training results, including things like aggregated loss and metrics.
                Tuple also contains the elapsed time in seconds for the round.
        """
        start_time = datetime.datetime.now()
        self.reports_manager.report(
            {
                "fit_start": str(start_time),
                "host_type": "server",
            }
        )

        self.update_before_fit(num_rounds, timeout)

        if self.per_round_checkpointer is not None:
            history, elapsed_time = self.fit_with_per_round_checkpointing(num_rounds, timeout)
        else:
            history, elapsed_time = super().fit(num_rounds, timeout)
        end_time = datetime.datetime.now()
        self.reports_manager.report(
            {
                "fit_elapsed_time": round((end_time - start_time).total_seconds()),
                "fit_end": str(end_time),
                "num_rounds": num_rounds,
                "host_type": "server",
            }
        )

        # WARNING: This will not work with wandb. Wandb reporting must be done live.
        self.report_centralized_eval(history, num_rounds)

        return history, elapsed_time

    def fit_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]]:
        """
        This function is called at each round of federated training. The flow is generally the same as a flower
        server, where clients are sampled and client side training is requested from the clients that are chosen.
        This function simply adds a bit of logging, post processing of the results

        Args:
            server_round (int): Current round number of the FL training. Begins at 1
            timeout (Optional[float]): Time that the server should wait (in seconds) for responses from the clients.
                Defaults to None, which indicates indefinite timeout.

        Returns:
            Optional[Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]]: The results of training
                on the client sit. The first set of parameters are the AGGREGATED parameters from the strategy. The
                second is a dictionary of AGGREGATED metrics. The third component holds the individual (non-aggregated)
                parameters, loss, and metrics for successful and unsuccessful client-side training.
        """

        round_start = datetime.datetime.now()
        fit_round_results = super().fit_round(server_round, timeout)
        round_end = datetime.datetime.now()

        self.reports_manager.report(
            {
                "fit_round_start": str(round_start),
                "fit_round_end": str(round_end),
                "fit_round_time_elapsed": round((round_end - round_start).total_seconds()),
            },
            server_round,
        )
        if fit_round_results is not None:
            _, metrics, fit_results_and_failures = fit_round_results
            self.reports_manager.report({"fit_round_metrics": metrics}, server_round)
            failures = fit_results_and_failures[1] if fit_results_and_failures else None

            if failures and not self.accept_failures:
                self._log_client_failures(failures)
                self._terminate_after_unacceptable_failures(timeout)

        return fit_round_results

    def shutdown(self) -> None:
        """
        Currently just records termination of the server process and disconnects and reporters that need to be.
        """
        self.reports_manager.report({"shutdown": str(datetime.datetime.now())})
        self.reports_manager.shutdown()

    def poll_clients_for_sample_counts(self, timeout: Optional[float]) -> List[int]:
        """
        Poll clients for sample counts from their training set, if you want to use this functionality your strategy
        needs to inherit from the StrategyWithPolling ABC and implement a configure_poll function.

        Args:
            timeout (Optional[float]): Timeout for how long the server will wait for clients to report counts. If none
                then the server waits indefinitely.

        Returns:
            List[int]: The number of training samples held by each client in the pool of available clients.
        """
        # Poll clients for sample counts, if you want to use this functionality your strategy needs to inherit from
        # the StrategyWithPolling ABC and implement a configure_poll function
        log(INFO, "Polling Clients for sample counts")
        assert isinstance(self.strategy, StrategyWithPolling)
        client_instructions = self.strategy.configure_poll(server_round=1, client_manager=self._client_manager)
        results, _ = poll_clients(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
        )

        sample_counts: List[int] = [
            int(get_properties_res.properties["num_train_samples"]) for (_, get_properties_res) in results
        ]
        log(INFO, f"Polling complete: Retrieved {len(sample_counts)} sample counts")

        return sample_counts

    def evaluate_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[Tuple[Optional[float], Dict[str, Scalar], EvaluateResultsAndFailures]]:
        # By default the checkpointing works off of the aggregated evaluation loss from each of the clients
        # NOTE: parameter aggregation occurs **before** evaluation, so the parameters held by the server have been
        # updated prior to this function being called.
        start_time = datetime.datetime.now()
        eval_round_results = self._evaluate_round(server_round, timeout)
        end_time = datetime.datetime.now()
        if eval_round_results:
            loss_aggregated, metrics_aggregated, (_, failures) = eval_round_results

            if failures and not self.accept_failures:
                self._log_client_failures(failures)
                self._terminate_after_unacceptable_failures(timeout)

            if loss_aggregated:
                self._maybe_checkpoint(loss_aggregated, metrics_aggregated, server_round)
                # Report evaluation results
                report_data = {
                    "val - loss - aggregated": loss_aggregated,
                    "round": server_round,
                    "eval_round_start": str(start_time),
                    "eval_round_end": str(end_time),
                    "eval_round_time_elapsed": round((end_time - start_time).total_seconds()),
                }

                if self.fl_config.get("local_epochs", None) is not None:
                    report_data["fit_epoch"] = server_round * self.fl_config["local_epochs"]
                elif self.fl_config.get("local_steps", None) is not None:
                    report_data["fit_step"] = server_round * self.fl_config["local_steps"]
                self.reports_manager.report(report_data, server_round)
                if len(metrics_aggregated) > 0:
                    self.reports_manager.report(
                        {"eval_round_metrics_aggregated": metrics_aggregated},
                        server_round,
                    )

        return eval_round_results

    def _log_fl_config(self) -> None:
        log(INFO, "FL Configuration:") if self.fl_config else log(INFO, "FL Config is Empty")
        for config_key, config_value in self.fl_config.items():
            if not isinstance(config_value, bytes):
                log(INFO, f"Key: {config_key} Value: {config_value!r}")

    def _hydrate_model_for_checkpointing(self) -> None:
        """
        This function is used as a means of saving the server-side model after aggregation in the FL training
        trajectory. In the current implementation, the server only holds numpy arrays. Without knowledge of a model
        architecture to which the arrays correspond. Thus, in the default implementation, we require that a torch
        architecture have been provided (self.server_model) and a parameter exchanger (self.parameter_exchanger) be
        provided which handles mapping these numpy arrays into the architecture properly.

        This function may be overridden if different behavior is desired.

        NOTE: This function stores the weights directly in the self.server_model attribute
        """

        assert self.server_model is not None, (
            "Model hydration has been called but no server_model is defined to hydrate. The functionality of "
            "_hydrate_model_for_checkpointing can be overridden if checkpointing without a torch architecture is "
            "possible and desired"
        )
        assert self.parameter_exchanger is not None, (
            "Model hydration has been called but no parameter_exchanger is defined to hydrate. The functionality of "
            "_hydrate_model_for_checkpointing can be overridden if checkpointing without a parameter exchanger is "
            "possible and desired"
        )
        model_ndarrays = parameters_to_ndarrays(self.parameters)
        self.parameter_exchanger.pull_parameters(model_ndarrays, self.server_model)

    def _save_server_state(self) -> None:
        """
        Save server checkpoint consisting of model, history, server round, metrics reporter and server name. This
        method can be overridden to add any necessary state to the checkpoint.
        """

        assert self.per_round_checkpointer is not None

        ckpt = {
            "model": self.server_model,
            "history": self.history,
            "current_round": self.current_round,
            "reports_manager": self.reports_manager,
            "server_name": self.server_name,
        }

        self.per_round_checkpointer.save_checkpoint(ckpt)

        log(INFO, f"Saving server state to checkpoint at {self.per_round_checkpointer.checkpoint_path}")

    def _load_server_state(self) -> None:
        """
        Load server checkpoint consisting of model, history, server name, current round and metrics reporter.
        The method can be overridden to add any necessary state when loading the checkpoint.
        """
        assert self.per_round_checkpointer is not None and self.per_round_checkpointer.checkpoint_exists()

        ckpt = self.per_round_checkpointer.load_checkpoint()

        log(INFO, f"Loading server state from checkpoint at {self.per_round_checkpointer.checkpoint_path}")

        narrow_dict_type_and_set_attribute(self, ckpt, "server_name", "server_name", str)
        narrow_dict_type_and_set_attribute(self, ckpt, "current_round", "current_round", int)
        narrow_dict_type_and_set_attribute(self, ckpt, "reports_manager", "reports_manager", ReportsManager)
        narrow_dict_type_and_set_attribute(self, ckpt, "history", "history", History)
        narrow_dict_type_and_set_attribute(self, ckpt, "model", "parameters", nn.Module, func=get_all_model_parameters)
        # Needed for when _hydrate_model_for_checkpointing is called
        narrow_dict_type_and_set_attribute(self, ckpt, "model", "server_model", nn.Module)

        self.parameters = get_all_model_parameters(ckpt["model"])

    def _terminate_after_unacceptable_failures(self, timeout: Optional[float]) -> None:
        assert not self.accept_failures
        # First we shutdown all clients involved in the FL training/evaluation if they can be.
        self.disconnect_all_clients(timeout=timeout)
        # Throw an exception alerting the user to failures on the client-side causing termination
        self.shutdown()
        raise ValueError(
            f"The server encountered failures from the clients and accept_failures is set to {self.accept_failures}"
        )

    def _log_client_failures(self, failures: FitFailures | EvaluateFailures) -> None:
        log(
            ERROR,
            f"There were {len(failures)} failures in the fitting process. This will result in termination of "
            "the FL process",
        )
        for failure in failures:
            if isinstance(failure, BaseException):
                log(
                    ERROR,
                    "An exception was returned instead of any failed results. As such the client ID is unknown. "
                    "Please check the client logs to determine which failed.\n"
                    f"The exception thrown was {repr(failure)}",
                )
            else:
                client_proxy, _ = failure
                log(
                    ERROR,
                    f"Client {client_proxy.cid} failed but did not return an exception. Partial results were received",
                )

    def _maybe_checkpoint(
        self,
        loss_aggregated: float,
        metrics_aggregated: Dict[str, Scalar],
        server_round: int,
    ) -> None:
        """
        This function will run through any provided checkpointers to save the server-side model. If no checkpointers
        are present, this function simply logs that no server-side checkpointing is performed.

        NOTE: The proper components for model hydration need to be in place for this implementation. If they are
        not an exception will be thrown.

        Args:
            loss_aggregated (float): aggregated loss value that can be used to determine whether to checkpoint
            metrics_aggregated (Dict[str, Scalar]): aggregated metrics from each of the clients for checkpointing
            server_round (int): What round of federated training we're on. This is just for logging purposes.
        """
        if self.checkpointer:
            self._hydrate_model_for_checkpointing()
            assert self.server_model is not None
            for checkpointer in self.checkpointer:
                checkpointer.maybe_checkpoint(self.server_model, loss_aggregated, metrics_aggregated)
        elif server_round == 1:
            # No checkpointer, just log message on the first round
            log(
                INFO,
                "No checkpointer present. Models will not be checkpointed on server-side.",
            )

    def _get_initial_parameters(self, server_round: int, timeout: Optional[float]) -> Parameters:
        """
        Get initial parameters from one of the available clients. This function is the same as the parent function
        in the flower server class except that we make use of the on_parameter_initialization_config_fn to provide
        a non-empty config to a client when requesting parameters from which to initialize all other clients.

        NOTE: The default behavior of flower servers is to simply send over a blank config, but this is insufficient
        for certain uses, where the client requires additional information from the server. This is needed, for example
        in nnUnet based Servers. An issue has been logged with flower: https://github.com/adap/flower/issues/3770
        """
        # Server-side parameter initialization
        parameters: Optional[Parameters] = self.strategy.initialize_parameters(client_manager=self._client_manager)
        if parameters is not None:
            log(INFO, "Using initial global parameters provided by strategy")
            return parameters

        # Get initial parameters from one of the clients
        log(INFO, "Requesting initial parameters from one random client")
        random_client = self._client_manager.sample(1)[0]
        if self.on_init_parameters_config_fn is None:
            # An empty configuration is the default for Flower servers
            ins = GetParametersIns(config={})
        else:
            ins = GetParametersIns(config=self.on_init_parameters_config_fn(server_round))
        get_parameters_res = random_client.get_parameters(ins=ins, timeout=timeout, group_id=server_round)
        if get_parameters_res.status.code == Code.OK:
            log(INFO, "Received initial parameters from one random client")
        else:
            log(
                WARNING,
                "Failed to receive initial parameters from the client. Empty initial parameters will be used.",
            )
        return get_parameters_res.parameters

    def _unpack_metrics(
        self, results: List[Tuple[ClientProxy, EvaluateRes]]
    ) -> Tuple[List[Tuple[ClientProxy, EvaluateRes]], List[Tuple[ClientProxy, EvaluateRes]]]:
        val_results = []
        test_results = []

        for client_proxy, eval_res in results:
            val_metrics = {
                k: v for k, v in eval_res.metrics.items() if not k.startswith(MetricPrefix.TEST_PREFIX.value)
            }
            test_metrics = {k: v for k, v in eval_res.metrics.items() if k.startswith(MetricPrefix.TEST_PREFIX.value)}

            if len(test_metrics) > 0:
                assert TEST_LOSS_KEY in test_metrics and TEST_NUM_EXAMPLES_KEY in test_metrics, (
                    f"'{TEST_NUM_EXAMPLES_KEY}' and '{TEST_LOSS_KEY}' keys must be present in "
                    "test_metrics dictionary for aggregation"
                )
                # Remove loss and num_examples from test_metrics if they exist
                test_loss = float(test_metrics.pop(TEST_LOSS_KEY))
                test_num_examples = int(test_metrics.pop(TEST_NUM_EXAMPLES_KEY))
                test_eval_res = EvaluateRes(eval_res.status, test_loss, test_num_examples, test_metrics)
                test_results.append((client_proxy, test_eval_res))

            val_eval_res = EvaluateRes(eval_res.status, eval_res.loss, eval_res.num_examples, val_metrics)
            val_results.append((client_proxy, val_eval_res))

        return val_results, test_results

    def _handle_result_aggregation(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        val_results, test_results = self._unpack_metrics(results)

        # Aggregate the validation results
        val_aggregated_result: Tuple[
            Optional[float],
            Dict[str, Scalar],
        ] = self.strategy.aggregate_evaluate(server_round, val_results, failures)
        val_loss_aggregated, val_metrics_aggregated = val_aggregated_result

        # Aggregate the test results if they are present
        if len(test_results) > 0:
            test_aggregated_result: Tuple[
                Optional[float],
                Dict[str, Scalar],
            ] = self.strategy.aggregate_evaluate(server_round, test_results, failures)
            test_loss_aggregated, test_metrics_aggregated = test_aggregated_result

            for key, value in test_metrics_aggregated.items():
                val_metrics_aggregated[key] = value
            if test_loss_aggregated is not None:
                val_metrics_aggregated[f"{MetricPrefix.TEST_PREFIX.value} loss - aggregated"] = test_loss_aggregated

        return val_loss_aggregated, val_metrics_aggregated

    def _evaluate_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[Tuple[Optional[float], Dict[str, Scalar], EvaluateResultsAndFailures]]:
        """Validate current global model on a number of clients."""
        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_evaluate(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self._client_manager,
        )
        if not client_instructions:
            log(INFO, "evaluate_round %s: no clients selected, cancel", server_round)
            return None
        log(
            DEBUG,
            "evaluate_round %s: strategy sampled %s clients (out of %s)",
            server_round,
            len(client_instructions),
            self._client_manager.num_available(),
        )
        # Collect `evaluate` results from all clients participating in this round
        # flwr sets group_id to server_round by default, so we follow that convention
        results, failures = evaluate_clients(
            client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
            group_id=server_round,
        )
        log(
            DEBUG,
            "evaluate_round %s received %s results and %s failures for Validation",
            server_round,
            len(results),
            len(failures),
        )

        val_loss_aggregated, val_metrics_aggregated = self._handle_result_aggregation(server_round, results, failures)

        return val_loss_aggregated, val_metrics_aggregated, (results, failures)
