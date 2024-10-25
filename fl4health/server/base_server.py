import datetime
from logging import DEBUG, INFO, WARN, WARNING
from pathlib import Path
from typing import Dict, Generic, List, Optional, Sequence, Tuple, TypeVar, Union

import torch.nn as nn
from flwr.common import EvaluateRes, Parameters
from flwr.common.logger import log
from flwr.common.parameter import parameters_to_ndarrays
from flwr.common.typing import Code, GetParametersIns, Scalar
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.history import History
from flwr.server.server import EvaluateResultsAndFailures, FitResultsAndFailures, Server, evaluate_clients
from flwr.server.strategy import Strategy

from fl4health.checkpointing.checkpointer import PerRoundCheckpointer, TorchCheckpointer
from fl4health.parameter_exchange.parameter_exchanger_base import ParameterExchanger
from fl4health.reporting.base_reporter import BaseReporter
from fl4health.reporting.reports_manager import ReportsManager
from fl4health.server.polling import poll_clients
from fl4health.strategies.strategy_with_poll import StrategyWithPolling
from fl4health.utils.config import narrow_dict_type_and_set_attribute
from fl4health.utils.metrics import TEST_LOSS_KEY, TEST_NUM_EXAMPLES_KEY, TestMetricPrefix
from fl4health.utils.parameter_extraction import get_all_model_parameters
from fl4health.utils.random import generate_hash


class FlServer(Server):
    def __init__(
        self,
        client_manager: ClientManager,
        strategy: Optional[Strategy] = None,
        reporters: Sequence[BaseReporter] | None = None,
        checkpointer: Optional[Union[TorchCheckpointer, Sequence[TorchCheckpointer]]] = None,
        server_name: Optional[str] = None,
    ) -> None:
        """
        Base Server for the library to facilitate strapping additional/useful machinery to the base flwr server.

        Args:
            client_manager (ClientManager): Determines the mechanism by which clients
                are sampled by the server, if they are to be sampled at all.
            strategy (Optional[Strategy], optional): The aggregation strategy to be
                used by the server to handle. client updates and other information
                potentially sent by the participating clients. If None the strategy is
                FedAvg as set by the flwr Server.
            reporters (Sequence[BaseReporter], optional): A sequence of FL4Health
                reporters which the server should send data to before and after each round.
            checkpointer (TorchCheckpointer | Sequence [TorchCheckpointer], optional):
                To be provided if the server should perform server side checkpointing
                based on some criteria. If none, then no server-side checkpointing is
                performed. Multiple checkpointers can also be passed in a sequence to
                checkpointer based on multiple criteria. Ensure checkpoint names are
                different for each checkpoint or they will overwrite on another.
                Defaults to None.
            server_name (Optional[str]): An optional string name to uniquely identify
                server.
        """

        super().__init__(client_manager=client_manager, strategy=strategy)
        self.checkpointer = [checkpointer] if isinstance(checkpointer, TorchCheckpointer) else checkpointer
        self.server_name = server_name if server_name is not None else generate_hash()

        # Initialize reporters with server name information.
        self.reports_manager = ReportsManager(reporters)
        self.reports_manager.initialize(id=self.server_name)

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
            self.reports_manager.report({"eval_metrics_centralized": round_metrics}, round + 1)

    def fit(self, num_rounds: int, timeout: Optional[float]) -> Tuple[History, float]:
        """
        Run federated learning for a number of rounds.

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
        history, elapsed_time = super().fit(num_rounds, timeout)
        end_time = datetime.datetime.now()
        self.reports_manager.report(
            {
                "fit_elapsed_time": str(start_time - end_time),
                "fit_end": str(end_time),
                "num_rounds": num_rounds,
                "host_type": "server",
            }
        )

        self.report_centralized_eval(history, num_rounds)

        return history, elapsed_time

    def fit_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]]:
        round_start = datetime.datetime.now()
        fit_round_results = super().fit_round(server_round, timeout)
        round_end = datetime.datetime.now()

        self.reports_manager.report(
            {"fit_round_start": str(round_start), "fit_round_end": str(round_end)},
            server_round,
        )
        if fit_round_results is not None:
            _, metrics, _ = fit_round_results
            self.reports_manager.report({"fit_metrics": metrics}, server_round)

        return fit_round_results

    def shutdown(self) -> None:
        self.reports_manager.report({"shutdown": str(datetime.datetime.now())})
        self.reports_manager.shutdown()

    def _hydrate_model_for_checkpointing(self) -> nn.Module:
        """
        This function is used for converting server parameters into a torch model that can be checkpointed. Note that
        if an inheriting class wants to do server-side checkpointing this functionality needs to be defined there.

        Raises:
            NotImplementedError: If this is called by a child class and the behavior is not defined, we throw an error.

        Returns:
            nn.Module: Should return a torch model to be checkpointed by a torch checkpointer.
        """
        # This function is used for converting server parameters into a torch model that can be checkpointed
        raise NotImplementedError()

    def _maybe_checkpoint(
        self,
        loss_aggregated: float,
        metrics_aggregated: Dict[str, Scalar],
        server_round: int,
    ) -> None:
        if self.checkpointer:
            try:
                model = self._hydrate_model_for_checkpointing()
                for checkpointer in self.checkpointer:
                    checkpointer.maybe_checkpoint(model, loss_aggregated, metrics_aggregated)
            except NotImplementedError:
                # Checkpointer is defined but there is no server-side model hydration to produce a model from the
                # server state. This is not a deal breaker, but may be unintended behavior and the user will be warned
                if server_round == 1:
                    # just log message on the first round
                    log(
                        WARNING,
                        "Server model hydration is not defined but checkpointer is defined. Not checkpointing "
                        "model. Please ensure that this is intended",
                    )
        elif server_round == 1:
            # No checkpointer, just log message on the first round
            log(
                INFO,
                "No checkpointer present. Models will not be checkpointed on server-side.",
            )

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

    def _unpack_metrics(
        self, results: List[Tuple[ClientProxy, EvaluateRes]]
    ) -> Tuple[List[Tuple[ClientProxy, EvaluateRes]], List[Tuple[ClientProxy, EvaluateRes]]]:
        val_results = []
        test_results = []

        for client_proxy, eval_res in results:
            val_metrics = {
                k: v for k, v in eval_res.metrics.items() if not k.startswith(TestMetricPrefix.TEST_PREFIX.value)
            }
            test_metrics = {
                k: v for k, v in eval_res.metrics.items() if k.startswith(TestMetricPrefix.TEST_PREFIX.value)
            }

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
                val_metrics_aggregated[f"{TestMetricPrefix.TEST_PREFIX.value} loss - aggregated"] = (
                    test_loss_aggregated
                )

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
            loss_aggregated, metrics_aggregated, _ = eval_round_results
            if loss_aggregated:
                self._maybe_checkpoint(loss_aggregated, metrics_aggregated, server_round)
                # Report evaluation results
                self.reports_manager.report(
                    {
                        "val - loss - aggregated": loss_aggregated,
                        "round": server_round,
                        "eval_round_start": str(start_time),
                        "eval_round_end": str(end_time),
                    },
                    server_round,
                )
                if len(metrics_aggregated) > 0:
                    self.reports_manager.report(
                        {"eval_metrics_aggregated": metrics_aggregated},
                        server_round,
                    )

        return eval_round_results


ExchangerType = TypeVar("ExchangerType", bound=ParameterExchanger)


class FlServerWithCheckpointing(FlServer, Generic[ExchangerType]):
    def __init__(
        self,
        client_manager: ClientManager,
        parameter_exchanger: ExchangerType,
        model: Optional[nn.Module] = None,
        strategy: Optional[Strategy] = None,
        reporters: Sequence[BaseReporter] | None = None,
        checkpointer: Optional[Union[TorchCheckpointer, Sequence[TorchCheckpointer]]] = None,
        intermediate_server_state_dir: Optional[Path] = None,
        server_name: Optional[str] = None,
    ) -> None:
        """
        This is a standard FL server but equipped with the assumption that the parameter exchanger is capable of
        hydrating the provided server model fully such that it can be checkpointed. For custom checkpointing
        functionality, one need only override _hydrate_model_for_checkpointing. If intermediate_server_state_dir
        is not None, performs per round checkpointing.


        Args:
            client_manager (ClientManager): Determines the mechanism by which clients are sampled by the server, if
                they are to be sampled at all.
            parameter_exchanger (ExchangerType): This is the parameter exchanger to be used to hydrate the model.
            model (Optional[nn.Module]): This is the torch model to be hydrated
                by the _hydrate_model_for_checkpointing function. Defaults to None.
            strategy (Optional[Strategy], optional): The aggregation strategy to be used by the server to handle
                client updates and other information potentially sent by the participating clients. If None the
                strategy is FedAvg as set by the flwr Server.
            reporters (Sequence[BaseReporter], optional): A sequence of FL4Health
                reporters which the server should send data to before and after each round.
            checkpointer (Optional[Union[TorchCheckpointer, Sequence[TorchCheckpointer]]], optional):
                To be provided if the server should perform server side checkpointing
                based on some criteria. If none, then no server-side checkpointing is performed. Multiple checkpointers
                can also be passed in a sequence to checkpoint based on multiple criteria. Defaults to None.
            intermediate_server_state_dir (Path): A directory to store and load checkpoints from for the server
                during an FL experiment.
            server_name (Optional[str]): An optional string name to uniquely identify server.
        """
        super().__init__(
            client_manager,
            strategy,
            reporters,
            checkpointer,
            server_name=server_name,
        )
        self.server_model = model
        # To facilitate model rehydration from server-side state for checkpointing
        self.parameter_exchanger = parameter_exchanger

        self.per_round_checkpointer: Union[None, PerRoundCheckpointer]

        if intermediate_server_state_dir is not None:
            log(
                WARNING,
                "intermediate_server_state_dir is not None. Creating PerRoundCheckpointer. \
                This functionality still experimental and only supported for BasicClient and NnunetClient currently.",
            )
            self.per_round_checkpointer = PerRoundCheckpointer(
                intermediate_server_state_dir, Path(f"{self.server_name}.ckpt")
            )
        else:
            self.per_round_checkpointer = None

        self.current_round: int
        self.history: History

    def _hydrate_model_for_checkpointing(self) -> nn.Module:
        assert (
            self.server_model is not None
        ), "Model hydration has been called but no server_model is defined to hydrate"
        model_ndarrays = parameters_to_ndarrays(self.parameters)
        self.parameter_exchanger.pull_parameters(model_ndarrays, self.server_model)
        return self.server_model

    def fit(self, num_rounds: int, timeout: Optional[float]) -> Tuple[History, float]:
        """
        Overrides method in parent class to call custom fit_with_per_round_checkpointing if an
        intermediate_server_state_dir is provided. Otherwise calls standard fit method.

        Args:
            num_rounds (int): The number of rounds to perform federated learning.
            timeout (Optional[float]): The timeout for clients to return results in a given FL round.

        Returns:
            Tuple[History, float]: The first element of the tuple is a history object containing the losses and
                metrics computed during training and validation. The second element of the tuple is
                the elapsed time in seconds.
        """
        if self.per_round_checkpointer is not None:
            start_time = datetime.datetime.now()
            self.reports_manager.report(
                {
                    "fit_start": str(start_time),
                    "host_type": "server",
                }
            )
            history, elapsed_time = self.fit_with_per_epoch_checkpointing(num_rounds, timeout)
            end_time = datetime.datetime.now()
            self.reports_manager.report(
                {
                    "fit_elapsed_time": str(start_time - end_time),
                    "fit_end": str(end_time),
                    "num_rounds": num_rounds,
                    "host_type": "server",
                }
            )
        else:
            # parent method includes call to report metrics to wandb if specified
            history, elapsed_time = super().fit(num_rounds, timeout)

        return history, elapsed_time

    def fit_with_per_epoch_checkpointing(self, num_rounds: int, timeout: Optional[float]) -> Tuple[History, float]:
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
            self.load_server_state()
        else:
            log(INFO, "Initializing server state")
            self.parameters = self._get_initial_parameters(server_round=1, timeout=timeout)
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
            self.save_server_state()

        # Bookkeeping
        end_time = datetime.datetime.now()
        elapsed_time = end_time - start_time
        log(INFO, "FL finished in %s", str(elapsed_time))
        return self.history, elapsed_time.total_seconds()

    def save_server_state(self) -> None:
        """
        Save server checkpoint consisting of model, history, server round, metrics reporter and server name.
            This method can be overridden to add any necessary state to the checkpoint.
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

        log(
            INFO,
            f"Saving server state to checkpoint at {self.per_round_checkpointer.checkpoint_path}",
        )

    def load_server_state(self) -> None:
        """
        Load server checkpoint consisting of model, history, server name, current round and metrics reporter.
            The method can be overridden to add any necessary state when loading the checkpoint.
        """
        assert self.per_round_checkpointer is not None and self.per_round_checkpointer.checkpoint_exists()

        ckpt = self.per_round_checkpointer.load_checkpoint()

        log(
            INFO,
            f"Loading server state from checkpoint at {self.per_round_checkpointer.checkpoint_path}",
        )

        narrow_dict_type_and_set_attribute(self, ckpt, "server_name", "server_name", str)
        narrow_dict_type_and_set_attribute(self, ckpt, "current_round", "current_round", int)
        narrow_dict_type_and_set_attribute(self, ckpt, "reports_manager", "reports_manager", ReportsManager)
        narrow_dict_type_and_set_attribute(self, ckpt, "history", "history", History)
        narrow_dict_type_and_set_attribute(self, ckpt, "model", "parameters", nn.Module, func=get_all_model_parameters)
        # Needed for when _hydrate_model_for_checkpointing is called
        narrow_dict_type_and_set_attribute(self, ckpt, "model", "server_model", nn.Module)

        self.parameters = get_all_model_parameters(ckpt["model"])


class FlServerWithInitializer(FlServer):
    def __init__(
        self,
        client_manager: ClientManager,
        strategy: Optional[Strategy] = None,
        reporters: Sequence[BaseReporter] | None = None,
        checkpointer: Optional[Union[TorchCheckpointer, Sequence[TorchCheckpointer]]] = None,
    ) -> None:
        """
        Server with an initialize hook method that is called prior to fit. Override the self.initialize method to do
        server initialization prior to training but after the clients have been created. Can be useful if the state of
        the server depends on the properties of the clients. Eg. The nnunet server requests an nnunet plans dict to be
        generated by a client if one was not provided.

        Args:
            client_manager (ClientManager): Determines the mechanism by which clients are sampled by the server, if
                they are to be sampled at all.
            strategy (Optional[Strategy], optional): The aggregation strategy to be used by the server to handle.
                client updates and other information potentially sent by the participating clients. If None the
                strategy is FedAvg as set by the flwr Server.
            reporters (Sequence[BaseReporter], optional): A sequence of FL4Health
                reporters which the server should send data to before and after each round.
            checkpointer (Optional[Union[TorchCheckpointer, Sequence
                [TorchCheckpointer]]], optional): To be provided if the server
                should perform server side checkpointing based on some
                criteria. If none, then no server-side checkpointing is
                performed. Defaults to None.
        """
        super().__init__(client_manager, strategy, reporters, checkpointer)
        self.initialized = False

    def _get_initial_parameters(self, server_round: int, timeout: Optional[float]) -> Parameters:
        """
        Get initial parameters from one of the available clients. Same as
        parent function except we provide a config to the client when
        requesting initial parameters
        https://github.com/adap/flower/issues/3770

        Note:
            I have to use configure_fit to bypass mypy errors since
            on_fit_config is not defined in the Strategy base class. The
            downside is that configure fit will wait until enough clients for
            training are present instead of just sampling one client. I
            thought about defining a new init_config attribute but this
        """
        # Server-side parameter initialization
        parameters: Optional[Parameters] = self.strategy.initialize_parameters(client_manager=self._client_manager)
        if parameters is not None:
            log(INFO, "Using initial global parameters provided by strategy")
            return parameters

        # Get initial parameters from one of the clients
        log(INFO, "Requesting initial parameters from one random client")
        random_client = self._client_manager.sample(1)[0]
        dummy_params = Parameters([], "None")
        config = self.strategy.configure_fit(server_round, dummy_params, self._client_manager)[0][1].config
        ins = GetParametersIns(config=config)
        get_parameters_res = random_client.get_parameters(ins=ins, timeout=timeout, group_id=server_round)
        if get_parameters_res.status.code == Code.OK:
            log(INFO, "Received initial parameters from one random client")
        else:
            log(
                WARN,
                "Failed to receive initial parameters from the client." " Empty initial parameters will be used.",
            )
        return get_parameters_res.parameters

    def initialize(self, server_round: int, timeout: Optional[float] = None) -> None:
        """
        Hook method to allow the server to do some additional initialization
        prior to training. For example, NnUNetServer uses this method to ask a
        client to initialize the global nnunet plans if one is not provided in
        in the config

        Args:
            server_round (int): The current server round. This hook method is
                only called with a server_round=0 at the beginning of self.fit
            timeout (Optional[float], optional): The server's timeout
                parameter. Useful if one is requesting information from a
                client Defaults to None.
        """
        self.initialized = True

    def fit(self, num_rounds: int, timeout: Optional[float]) -> Tuple[History, float]:
        """
        Same as parent method except initialize hook method is called first
        """
        # Initialize the server
        if not self.initialized:
            self.initialize(server_round=0, timeout=timeout)

        return super().fit(num_rounds, timeout)
