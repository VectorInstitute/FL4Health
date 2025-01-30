import argparse
import os
from functools import partial
from logging import INFO
from pathlib import Path
from typing import Any

import flwr as fl
from flwr.common.logger import log
from flwr.common.typing import Config, Metrics
from flwr.server.client_manager import SimpleClientManager
from flwr.server.strategy import FedAdam
from sklearn.model_selection import train_test_split

from examples.fedopt_example.client_data import LabelEncoder, Vocabulary, get_local_data, word_tokenize
from examples.fedopt_example.metrics import Outcome, ServerMetrics
from examples.models.lstm_model import LSTM
from fl4health.servers.base_server import FlServer
from fl4health.utils.config import load_config
from fl4health.utils.parameter_extraction import get_all_model_parameters


def metric_aggregation(all_client_metrics: list[tuple[int, Metrics]]) -> Metrics:
    total_preds = 0
    true_preds = 0
    outcome_dict: dict[str, Outcome] = {}
    # Run through all of the metrics
    for _, client_metrics in all_client_metrics:
        for metric_name, metric_value in client_metrics.items():
            # If it's an overall count, we accumulate
            if metric_name == "total_preds":
                assert isinstance(metric_value, int)
                total_preds += metric_value
            elif metric_name == "true_preds":
                assert isinstance(metric_value, int)
                true_preds += metric_value
            # Otherwise it's class related and we handle parsing and aggregation through class functions.
            else:
                assert isinstance(metric_value, str)
                client_outcome = Outcome.from_results_dict(metric_name, metric_value)
                if metric_name in outcome_dict:
                    outcome_dict[metric_name] = Outcome.merge_outcomes(outcome_dict[metric_name], client_outcome)
                else:
                    outcome_dict[metric_name] = client_outcome

        server_metrics = ServerMetrics(true_preds, total_preds, list(outcome_dict.values()))

    return server_metrics.compute_metrics()


def construct_config(
    current_round: int,
    sequence_length: int,
    local_epochs: int,
    batch_size: int,
    vocab_dimension: int,
    hidden_size: int,
    vocabulary: Vocabulary,
    label_encoder: LabelEncoder,
) -> Config:
    # NOTE: The omitted variable is server_round which allows for dynamically changing the config each round
    return {
        "sequence_length": sequence_length,
        "local_epochs": local_epochs,
        "batch_size": batch_size,
        "vocab_dimension": vocab_dimension,
        "hidden_size": hidden_size,
        "vocabulary": vocabulary.to_json(),
        "label_encoder": label_encoder.to_json(),
        "current_server_round": current_round,
    }


def fit_config(
    sequence_length: int,
    local_epochs: int,
    batch_size: int,
    vocab_dimension: int,
    hidden_size: int,
    vocabulary: Vocabulary,
    label_encoder: LabelEncoder,
    current_round: int,
) -> Config:
    return construct_config(
        current_round,
        sequence_length,
        local_epochs,
        batch_size,
        vocab_dimension,
        hidden_size,
        vocabulary,
        label_encoder,
    )


def pretrain_vocabulary(path: Path) -> tuple[Vocabulary, LabelEncoder]:
    df = get_local_data(path)
    # Drop 20% of the texts to artificially create some UNK tokens
    processed_df, _ = train_test_split(df, test_size=0.8)
    headline_text = [word_tokenize(text.lower()) for _, text in processed_df["title"].items()]
    body_text = [word_tokenize(text.lower()) for _, text in processed_df["body"].items()]
    label_encoder = LabelEncoder.encoder_from_dataframe(processed_df, "category")
    return Vocabulary(None, headline_text + body_text), label_encoder


def main(config: dict[str, Any]) -> None:
    log(INFO, "Fitting vocabulary to a centralized text sample")
    data_path = Path(
        os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "datasets",
            "agnews_data",
            "partitioned_datasets",
            "partition_0.json",
        )
    )
    # Each of the clients needs a shared vocabulary and label encoder to produce their own data loaders
    vocabulary, label_encoder = pretrain_vocabulary(data_path)
    log(INFO, "Central vocabulary fitted")

    # This function will be used to produce a config that is sent to each client to initialize their own environment
    fit_config_fn = partial(
        fit_config,
        config["sequence_length"],
        config["local_epochs"],
        config["batch_size"],
        config["vocab_dimension"],
        config["hidden_size"],
        vocabulary,
        label_encoder,
    )

    initial_model = LSTM(vocabulary.vocabulary_size, config["vocab_dimension"], config["hidden_size"])

    # Server performs FedAdam as the server side optimization strategy.
    # Uses the default parameters for moment accumulation

    strategy = FedAdam(
        min_fit_clients=config["n_clients"],
        min_evaluate_clients=config["n_clients"],
        # Server waits for min_available_clients before starting FL rounds
        min_available_clients=config["n_clients"],
        fit_metrics_aggregation_fn=metric_aggregation,
        evaluate_metrics_aggregation_fn=metric_aggregation,
        on_fit_config_fn=fit_config_fn,
        # We use the same fit config function, as nothing changes for eval
        on_evaluate_config_fn=fit_config_fn,
        # Server side weight initialization
        initial_parameters=get_all_model_parameters(initial_model),
        accept_failures=False,
    )

    client_manager = SimpleClientManager()
    server = FlServer(client_manager=client_manager, fl_config=config, strategy=strategy, accept_failures=False)

    fl.server.start_server(
        server_address=config["server_address"],
        config=fl.server.ServerConfig(num_rounds=config["n_server_rounds"]),
        server=server,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Server Main")
    parser.add_argument(
        "--config_path",
        action="store",
        type=str,
        help="Path to configuration file.",
        default="examples/fedopt_example/config.yaml",
    )
    args = parser.parse_args()

    config = load_config(args.config_path)
    main(config)
