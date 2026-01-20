# Quickstart

## Installation

First, we need to install the `fl4health` package. The easiest and recommended
way to do this is via `pip`.

```sh
pip install fl4health
```

## A simple FL task

With federated learning, the model is trained collaboratively by a set of
distributed nodes called `clients`. This collaboration is facilitated by another
node, namely the `server` node. To setup an FL task we need to define our `Client`
as well as our `Server` in the scripts `client.py` and `server.py`, respectively.

### `client.py`

```python
from pathlib import Path

import flwr as fl
import torch
import torch.nn as nn
from flwr.common.typing import Config
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from examples.models.cnn_model import Net
from fl4health.clients.basic_client import BasicClient
from fl4health.utils.load_data import load_cifar10_data
from fl4health.metrics import Accuracy


class CifarClient(BasicClient):
    def get_data_loaders(self, config: Config) -> tuple[DataLoader, DataLoader]:
        train_loader, val_loader, _ = load_cifar10_data(self.data_path, batch_size=64)
        return train_loader, val_loader

    def get_criterion(self, config: Config) -> _Loss:
        return torch.nn.CrossEntropyLoss()

    def get_optimizer(self, config: Config) -> Optimizer:
        return torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

    def get_model(self, config: Config) -> nn.Module:
        return Net().to(self.device)


def main(dataset_path: str) -> None:
    client = CifarClient(data_path=Path(dataset_path), metrics=[Accuracy("accuracy")], device=torch.device("cpu"))
    fl.client.start_client(server_address="0.0.0.0:8080", client=client.to_client())
    client.shutdown()
```

### `server.py`

```python
from functools import partial

import flwr as fl
from flwr.common.typing import Config
from flwr.server.client_manager import SimpleClientManager
from flwr.server.strategy import FedAvg

from examples.models.cnn_model import Net
from fl4health.servers.base_server import FlServer
from fl4health.metrics.metric_aggregation import evaluate_metrics_aggregation_fn, fit_metrics_aggregation_fn
from fl4health.utils.parameter_extraction import get_all_model_parameters


def fit_config(current_server_round: int) -> Config:
    return {"local_epochs": 3, "batch_size": 64, "current_server_round": current_server_round}


def main() -> None:

    fit_config_fn = partial(fit_config)
    model = Net()
    strategy = FedAvg(
        min_fit_clients=2,
        min_evaluate_clients=2,
        # Server waits for min_available_clients before starting FL rounds
        min_available_clients=2,
        on_fit_config_fn=fit_config_fn,
        # We use the same fit config function, as nothing changes for eval
        on_evaluate_config_fn=fit_config_fn,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        initial_parameters=get_all_model_parameters(model),
    )
    server = FlServer(SimpleClientManager(), {}, strategy)

    fl.server.start_server(
        server=server,
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=20),
    )
```

## Running the FL task

Now that we have our server and clients defined, we can run the FL system!

### Starting Server

The next step is to start the server by running

```sh
python -m examples.basic_example.server
```

### Starting Clients

Once the server has started and logged "FL starting," the next step, in separate terminals, is to start the two
clients. This is done by simply running (remembering to activate your environment)

```sh
python -m examples.basic_example.client --dataset_path /path/to/data
```

**NOTE**: The argument `dataset_path` has two functions, depending on whether the dataset exists locally or not. If
the dataset already exists at the path specified, it will be loaded from there. Otherwise, the dataset will be
automatically downloaded to the path specified and used in the run.

After both clients have been started federated learning should commence.
