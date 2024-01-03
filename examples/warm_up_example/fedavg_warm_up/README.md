# FedAvg Warm Up Federated Learning Example
In this example, FedAvg is applied to a modified version of the MNIST dataset that is non--IID example as a warm up to other federated learning(FL) algorithms.
The FL server expects three clients to be spun up (i.e. it will wait until three clients report in before starting training). Each client has it's own modified version of the MNIST dataset. This modification essentially subsamples a certain number from the original training and validation sets of MNIST in order to synthetically induce local variations in the statistical properties of the clients training/validation data. In theory, the models should be able to perform well on their local data while learning from other clients data that has different statistical properties. The proportion of labels at each client is determined by dirichlet distribtuion across the classes. The lower the beta parameter is for each class, the higher the degree of the label heterogeneity.

The server has some custom metrics aggregation and uses Federated Averaging as its server-side optimization.

As this is a warm up training for a consecutive run with different Federated Learning (FL) algorithm, it is crucial to set the fixed seed for both clients and server to ensure uniformity in random data points across consecutive runs. Therefore, we make sure to set a fixed seed for these consecutive runs in both the `client.py` and `server.py` files. Additionally, it is important to assign a unique number to each client and establish a checkpointing strategy for the clients. This allows us to load each client's warmed-up model from this example in further examples. In this particular scenario, we set the checkpointing strategy to save the latest model and assigned a unique number between 1 and 3 to each client. This ensures that we can load the trained local model for each client from this example in subsequent runs as  warmed up model.

### Weights and Biases Reporting

This example is also capable of logging results to your Weights and Biases account by setting `enabled` to `True` in the `config.yaml` under the `reporting_config` section. You'll also need to set the `entity` value to your Weights and Biases entity. Once those two things are set, you should be able to run the example and log the results to W and B directly.

### Running the Example

In order to run the example, first ensure you have [installed the dependencies in your virtual environment according to the main README](/README.md#development-requirements) and it has been activated.

## Starting Server

The next step is to start the server by running
```
python -m examples.fedprox_example.server --config_path /path/to/config.yaml --seed "SEED"
```

from the FL4Health directory. The following arguments must be present in the specified config file:
* `n_clients`: number of clients the server waits for in order to run the FL training
* `local_epochs`: number of epochs each client will train for locally
* `batch_size`: size of the batches each client will train on
* `n_server_rounds`: The number of rounds to run FL

## Starting Clients

Once the server has started and logged "FL starting," the next step, in separate terminals, is to start the three
clients. This is done by simply running (remembering to activate your environment)
```
python3 -m examples.warm_up_example.fedavg_warm_up.client --dataset_path /path/to/data --seed "SEED" --checkpoint_dir /path/to/checkpointing/directory --client_number "CLIENT_NUMBER (1, 2, or 3)"
```

**NOTE**: The argument `dataset_path` has two functions, depending on whether the dataset exists locally or not. If
the dataset already exists at the path specified, it will be loaded from there. Otherwise, the dataset will be
automatically downloaded to the path specified and used in the run.

After all three clients have been started, federated learning should commence.
