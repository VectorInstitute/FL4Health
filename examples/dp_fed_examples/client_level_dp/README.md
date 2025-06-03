# Client Level Differential Privacy Federated Learning Example

This example shows how to implement Differential Privacy into the Federated Learning framework. In this case we focus on *client level* privacy which is a more substantial version of instance level DP, where the participation of an entire client's set of data is protected from training dataset membership inference. This example uses the FedAvgM implementation with unweighted averaging (To be implemented) suggested in Differentially Private Learning with Adaptive Clipping. The example uses an accountant specifically tailored to this approach. The clients are Poisson sampled by default.

## Running the Example
In order to run the example, first ensure you have [installed the dependencies in your virtual environment according to the main README](/README.md#development-requirements) and it has been activated.

## Starting Server

The next step is to start the server by running
```bash
python -m examples.dp_fed_examples.client_level_dp.server --config_path /path/to/config.yaml
```

## Starting Clients

Once the server has started and logged "FL starting," the next step, in separate terminals, is to start the clients expected by the server. This is done by simply running (remembering to activate your environment)
```bash
python -m examples.dp_fed_examples.client_level_dp.client --dataset_path /path/to/data
```
**NOTE**: The argument `dataset_path` has two functions, depending on whether the dataset exists locally or not. If
the dataset already exists at the path specified, it will be loaded from there. Otherwise, the dataset will be
automatically downloaded to the path specified and used in the run.

After the minimum number of clients have been started federated learning should commence.
