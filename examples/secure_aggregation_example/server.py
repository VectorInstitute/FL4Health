from fl4health.server.secure_aggregation_server import SecureAggregationServer
from flwr.server import start_server as RunServer
from flwr.server import ServerConfig
from argparse import ArgumentParser
from fl4health.utils.config import load_config
from flwr.server.client_manager import SimpleClientManager
from examples.models.cnn_model import Net
from fl4health.parameter_exchange.full_exchanger import FullParameterExchanger
from fl4health.checkpointing.checkpointer import BestMetricTorchCheckpointer
from examples.simple_metric_aggregation import evaluate_metrics_aggregation_fn, fit_metrics_aggregation_fn
from .utils import generate_config, get_parameters
from functools import partial
from fl4health.strategies.secure_aggregation_strategy import SecureAggregationStrategy

# replace later with secure aggregation strategy 
from fl4health.strategies.basic_fedavg import BasicFedAvg



if __name__ == '__main__':


    # get configurations from command line 
    parser = ArgumentParser(description="Secure aggregation server.")
    parser.add_argument(
        "--config_path",
        action="store",
        type=str,
        help="Path to configuration file. No enclosing quotes required.",
        default="examples/secure_aggregation_example/config.yaml",
    )
    args = parser.parse_args()
    config = load_config(args.config_path)

    # global model (server side)       
    model = Net()

    # consumed by strategy below
    config_parial = partial(generate_config, config["local_epochs"], config["batch_size"])

    strategy = SecureAggregationStrategy(
        min_fit_clients=config["n_clients"],
        min_evaluate_clients=config["n_clients"],
        min_available_clients=config["n_clients"],
        on_fit_config_fn=config_parial,
        on_evaluate_config_fn=config_parial,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        initial_parameters=get_parameters(model),
    )

    # configure server
    server = SecureAggregationServer(
        client_manager=SimpleClientManager(), 
        model=model, 
        parameter_exchanger=FullParameterExchanger(), 
        wandb_reporter=None, 
        strategy=strategy, 
        checkpointer=BestMetricTorchCheckpointer(config["checkpoint_path"], "best_model.pkl", maximize=False)
    )

    # run server
    RunServer(
        server=server,
        server_address="0.0.0.0:8080",
        config=ServerConfig(num_rounds=config["n_server_rounds"]),
    )