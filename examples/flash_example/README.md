# Flash Federated Learning Example

In this example, we implement the Flash algorithm from the paper [Flash: Concept Drift Adaptation in Federated Learning](https://proceedings.mlr.press/v202/panchal23a/panchal23a.pdf). This method enhances federated learning by addressing both statistical heterogeneity and concept drift issues.

### Running the Example

Ensure you have [installed the dependencies in your virtual environment according to the main README](/README.md#development-requirements) and it has been activated.

## Starting Server

Start the server by running:
```bash
python -m examples.flash_example.server --config_path /path/to/config.yaml
```
from the FL4Health directory. The config file should contain:
* `n_clients`: Number of clients the server waits for to run FL training.
* `local_epochs`: Number of epochs each client will train for locally.
* `batch_size`: Size of the batches each client will train on.
* `n_server_rounds`: Number of rounds to run FL.
* `gamma`: Early-stopping threshold for client-side training.

## Starting Clients

After the server starts and logs "FL starting," start the clients in separate terminals:
```bash
python -m examples.flash_example.client --dataset_path /path/to/data
```
**NOTE**: The `dataset_path` argument either loads the dataset if it exists locally or downloads it to the specified path.

Once all clients have started, federated learning will commence.
