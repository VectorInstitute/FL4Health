import argparse
from pathlib import Path
from typing import Any

import flwr as fl
import pandas as pd
from flwr.common.parameter import ndarrays_to_parameters
from flwr.common.typing import Parameters

from examples.models.mlp_classifier import MLP
from fl4health.client_managers.poisson_sampling_manager import PoissonSamplingClientManager
from fl4health.feature_alignment.tab_features_info_encoder import TabularFeaturesInfoEncoder
from fl4health.metrics.metric_aggregation import evaluate_metrics_aggregation_fn, fit_metrics_aggregation_fn
from fl4health.servers.tabular_feature_alignment_server import TabularFeatureAlignmentServer
from fl4health.strategies.basic_fedavg import BasicFedAvg
from fl4health.utils.config import load_config


# This data path is used to create a "source of truth" on the server-side as an example.
# This is used if the config specifies source_specified as true
DATA_PATH = "examples/feature_alignment_example/mimic3d_hospital1.csv"

CONFIG_PATH = "examples/feature_alignment_example/config.yaml"


def get_initial_model_parameters(input_dim: int, output_dim: int) -> Parameters:
    initial_model = MLP(input_dim, output_dim)
    return ndarrays_to_parameters([val.cpu().numpy() for _, val in initial_model.state_dict().items()])


def construct_tab_feature_info_encoder(
    data_path: Path, id_column: str, target_column: str
) -> TabularFeaturesInfoEncoder:
    df = pd.read_csv(data_path)
    return TabularFeaturesInfoEncoder.encoder_from_dataframe(df, id_column, target_column)


def main(config: dict[str, Any]) -> None:
    client_manager = PoissonSamplingClientManager()
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
        tabular_features_source_of_truth=tab_feature_info_encoder_hospital1,
        accept_failures=False,
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
