# Federated Evaluation Example
This example provides a simple implementation of a federated evaluation setup on the CIFAR dataset and two model checkpoints trained on that dataset. The federated evaluation server expects two clients to be spun up (i.e. it will wait until two clients report in before starting evaluation), as specified in the config.yaml under n_clients. Each client has the same "local" dataset. I.e. they each load the complete CIFAR test dataset and therefore have the same evaluation sets. The server performs uniform averaging of the client metrics, which are, in this case, accuracy and model loss. Note that since this is federated evaluation, no training occurs, therefore no strategy is implemented and no "server rounds" are processed. The server simply asks each client to evaluate a global and/or local model on the CIFAR test set and report the metrics back to the central server for aggregation. A global model checkpoint can be provided to the server, if one exists, representing a global model for all clients. It is passed through the parameter exchanger to the clients to be loaded. In `examples/assets` there are two checkpoint files representing two trained `Net()` models from `examples/models/cnn_model.py`. These models can be used to run this example by providing their relative path to the server and/or the clients.

## Running the Example
In order to run the example, first ensure you have [installed the dependencies in your virtual environment according to the main README](/README.md#development-requirements) and it has been activated.

## Starting Server

The next step is to start the server by running
```bash
python -m examples.federated_eval_example.server  --config_path "/path/to/config.yaml" --checkpoint_path "examples/assets/fed_eval_example/best_checkpoint_global.pkl"
```
from the FL4Health directory. The following arguments must be present in the specified config file:
* `n_clients`: number of clients the server waits for in order to run the FL training
* `local_epochs`: number of epochs each client will train for locally (not used in evaluation)
* `batch_size`: size of the batches each client will evaluate on
* `n_server_rounds`: The number of rounds to run FL (not used in evaluation)

## Starting Clients

Once the server has started and logged "FL starting," the next step, in separate terminals, is to start the two
clients. This is done by simply running (remembering to activate your environment)
```bash
python -m examples.federated_eval_example.client --dataset_path /path/to/data --checkpoint_path "examples/assets/fed_eval_example/best_checkpoint_local.pkl"
```
**NOTE**: The argument `dataset_path` has two functions, depending on whether the dataset exists locally or not. If
the dataset already exists at the path specified, it will be loaded from there. Otherwise, the dataset will be
automatically downloaded to the path specified and used in the run.

After both clients have been started federated evaluation should commence.
