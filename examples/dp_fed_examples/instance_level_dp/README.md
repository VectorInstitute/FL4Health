# Instance Level Differential Privacy Federated Learning Example

This example shows how to implement Differential Privacy into the Federated Learning framework. In this case we focus on *instance level* privacy rather than the more substantial client-level privacy. Hence, the example uses the Opacus DP-SGD algorithm to impose DP guarantees and accounting is done using instance-level privacy accountants. The server side optimization simply uses FedAvg to combine the weights at the server level. The clients and their underlying data are Poisson sampled.

## Running the Example
In order to run the example, first ensure you have [installed the dependencies in your virtual environment according to the main README](/README.md#development-requirements) and it has been activated.

## Starting Server

The next step is to start the server by running
```bash
python -m examples.dp_fed_examples.instance_level_dp.server --config_path /path/to/config.yaml
```

## Starting Clients

Once the server has started and logged "FL starting," the next step, in separate terminals, is to start the clients expected by the server. This is done by simply running (remembering to activate your environment)
```bash
python -m examples.dp_fed_examples.instance_level_dp.client --dataset_path /path/to/data
```
**NOTE**: The argument `dataset_path` has two functions, depending on whether the dataset exists locally or not. If
the dataset already exists at the path specified, it will be loaded from there. Otherwise, the dataset will be
automatically downloaded to the path specified and used in the run.

After the minimum number of clients have been started federated learning should commence.
