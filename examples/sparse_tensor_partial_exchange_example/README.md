# Sparse Tensor Federated Learning Example
This example demonstrates the usage of the sparse tensor parameter exchanger. At the end of each training round, each client selects an arbitrary set of its model parameters to be exchanged with the server.

On the server side, parameters belonging to the same tensor are first grouped together, then weighted averaging is performed in a per-tensor fashion.

The sparse tensor parameter exchanger can use various criteria to select the set of parameters to be exchanged. In this example, it selects those parameters that have the largest magnitude at the end of each training round. The number of parameters that are exchanged is determined by a sparsity level argument in the config file.

## Running the Example
In order to run the example, first ensure you have [installed the dependencies in your virtual environment according to the main README](/README.md#development-requirements) and it has been activated.

## Starting Server

The next step is to start the server by running
```bash
python -m examples.sparse_tensor_partial_exchange_example.server  --config_path /path/to/config.yaml
```
from the FL4Health directory. The following arguments must be present in the specified config file:
* `n_clients`: number of clients the server waits for in order to run the FL training
* `local_epochs`: number of epochs each client will train for locally
* `sparsity_level`: a real number between 0 and 1 that indicates the percentage of parameters that are exchanged between the server and clients.
* `batch_size`: size of the batches each client will train on
* `n_server_rounds`: The number of rounds to run FL

## Starting Clients

Once the server has started and logged "FL starting," the next step, in separate terminals, is to start the four
clients. This is done by simply running (remembering to activate your environment)
```bash
python -m examples.sparse_tensor_partial_exchange_example.client --dataset_path /path/to/data
```
**NOTE**: The argument `dataset_path` has two functions, depending on whether the dataset exists locally or not. If
the dataset already exists at the path specified, it will be loaded from there. Otherwise, the dataset will be
automatically downloaded to the path specified and used in the run.

After all four clients have been started federated learning should commence.
