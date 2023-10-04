import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import flwr as fl
import pandas as pd
from flwr.common.parameter import ndarrays_to_parameters
from flwr.common.typing import Metrics, Parameters

from examples.models.logistic_regression import LogisticRegression
from examples.simple_metric_aggregation import metric_aggregation, normalize_metrics
from fl4health.client_managers.poisson_sampling_manager import PoissonSamplingClientManager
from fl4health.feature_alignment.tab_features_info_encoder import TabFeaturesInfoEncoder
from fl4health.server.tabular_feature_alignment_server import TabularFeatureAlignmentServer
from fl4health.strategies.basic_fedavg import BasicFedAvg
from fl4health.utils.config import load_config

DATA_PATH = "examples/feature_alignment_example/mimic3d_hospital1.csv"
CONFIG_PATH = "examples/feature_alignment_example/config.yaml"


def get_initial_model_parameters(input_dim: int, output_dim: int) -> Parameters:
    # FedAdam requires that we provide server side parameter initialization.
    # Currently uses the Pytorch default initialization for the model parameters.
    initial_model = LogisticRegression(input_dim, output_dim)
    return ndarrays_to_parameters([val.cpu().numpy() for _, val in initial_model.state_dict().items()])


def fit_metrics_aggregation_fn(all_client_metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # This function is run by the server to aggregate metrics returned by each clients fit function
    # NOTE: The first value of the tuple is number of examples for FedAvg
    total_examples, aggregated_metrics = metric_aggregation(all_client_metrics)
    return normalize_metrics(total_examples, aggregated_metrics)


def evaluate_metrics_aggregation_fn(all_client_metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # This function is run by the server to aggregate metrics returned by each clients evaluate function
    # NOTE: The first value of the tuple is number of examples for FedAvg
    total_examples, aggregated_metrics = metric_aggregation(all_client_metrics)
    return normalize_metrics(total_examples, aggregated_metrics)


def construct_tab_feature_info_encoder(data_path: Path, id_column: str, target_column: str) -> TabFeaturesInfoEncoder:
    df = pd.read_csv(data_path)
    return TabFeaturesInfoEncoder.encoder_from_dataframe(df, id_column, target_column)


def main(config: Dict[str, Any]) -> None:
    client_manager = PoissonSamplingClientManager()
    # Server performs simple FedAveraging as its server-side optimization strategy
    strategy = BasicFedAvg(
        min_fit_clients=config["n_clients"],
        min_evaluate_clients=config["n_clients"],
        # Server waits for min_available_clients before starting FL rounds
        min_available_clients=config["n_clients"],
        on_fit_config_fn=None,
        on_evaluate_config_fn=None,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        initial_parameters=None,
    )

    source_specified = config["source_specified"]
    if source_specified:
        tab_feature_info_encoder_hospital1 = construct_tab_feature_info_encoder(
            Path(DATA_PATH), "hadm_id", "LOSgroupNum"
        )
    else:
        tab_feature_info_encoder_hospital1 = None

    server = TabularFeatureAlignmentServer(
        client_manager=client_manager,
        config=config,
        initialize_parameters=get_initial_model_parameters,
        strategy=strategy,
        tab_features_info=tab_feature_info_encoder_hospital1,
    )

    fl.server.start_server(
        server=server,
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=config["n_server_rounds"]),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Server Main")
    parser.add_argument(
        "--config_path",
        action="store",
        type=str,
        help="Path to configuration file.",
        default=CONFIG_PATH,
    )
    args = parser.parse_args()

    config = load_config(args.config_path)

    main(config)
