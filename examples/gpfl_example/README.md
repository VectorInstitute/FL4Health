# GPFL Example
This is an example implementation of the [GPFL: Simultaneously Learning Global and Personalized Feature Information for Personalized Federated Learning](https://arxiv.org/abs/2308.10279) algorithm on the MNIST dataset.
The FL server expects two clients to be spun up (i.e. it will wait until two clients report in before starting training). Each client has the same "local" dataset. I.e. they each load the complete MNIST dataset and therefore have the same training and validation sets. The server uses FedAvg to aggregate the model parameters shared by the clients. FedAvg is also the default server aggregation method used in GPFL paper.
Note that the model structure used by the clients and server should follow the model base defined in `fl4health.model_bases.gpfl_base`. Three optimizers must be defined in the client in a dictionary with the keys "model", "gce", and "cov", corresponding to the optimizers used for different sub-modules.

## Running the Example
In order to run the example, first ensure you have [installed the dependencies in your virtual environment according to the main README](/README.md#development-requirements) and it has been activated.

## Starting Server

The next step is to start the server by running
```bash
python -m examples.gpfl_example.server  --config_path /path/to/config.yaml
```
from the FL4Health directory. The following arguments must be present in the specified config file:
* `n_clients`: number of clients the server waits for in order to run the FL training
* `local_epochs`: number of epochs each client will train for locally
* `batch_size`: size of the batches each client will train on
* `n_server_rounds`: The number of rounds to run FL

## Starting Clients

Once the server has started and logged "FL starting," the next step, in separate terminals, is to start the two
clients. This is done by simply running (remembering to activate your environment)
```bash
python -m examples.gpfl_example.client --dataset_path /path/to/data --learning_rate 0.005 --mu 0.01 --lambda_parameter 0.01
```
**NOTE**: The argument `dataset_path` has two functions, depending on whether the dataset exists locally or not. If
the dataset already exists at the path specified, it will be loaded from there. Otherwise, the dataset will be
automatically downloaded to the path specified and used in the run.

After both clients have been started federated learning should commence.
