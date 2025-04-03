# FedAvg Warm Up Federated Learning Example
In this example, FedAvg is applied to a modified version of the MNIST dataset that is non-IID example as a warm up to other federated learning(FL) algorithms.
The FL server expects three clients to be spun up (i.e. it will wait until three clients report in before starting training). Each client has it's own modified version of the MNIST dataset. This modification essentially subsamples a certain number from the original training and validation sets of MNIST in order to synthetically induce local variations in the statistical properties of the clients training/validation data. In theory, the models should be able to perform well on their local data while learning from other clients data that has different statistical properties. The proportion of labels at each client is determined by Dirichlet distribution across the classes. The lower the beta parameter is for each class, the higher the degree of the label heterogeneity.

The server has some custom metrics aggregation and uses Federated Averaging as its server-side optimization.

As this is a warm-up training for consecutive runs with different Federated Learning (FL) algorithms, it is crucial to set a fixed seed for both clients and the server to ensure uniformity in random data points across these runs. Therefore, we make sure to set a fixed seed for these consecutive runs in both the `client.py` and `server.py` files. Additionally, it is important to establish a checkpointing strategy for the clients using their randomly generated unique client names. This allows us to load each client's warmed-up model from this example in further instances. In this particular scenario, we set the checkpointing strategy to save the latest model. This ensures that we can load the trained local model for each client from this example in subsequent runs as a warmed-up model.

### Running the Example

In order to run the example, first ensure you have [installed the dependencies in your virtual environment according to the main README](/README.md#development-requirements) and it has been activated.

## Starting Server

The next step is to start the server by running
```bash
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
```bash
python3 -m examples.warm_up_example.fedavg_warm_up.client --dataset_path /path/to/data --seed "SEED" --checkpoint_dir /path/to/checkpointing/directory --client_name "name_for_client"
```

**NOTE**: The argument `dataset_path` has two functions, depending on whether the dataset exists locally or not. If the dataset already exists at the path specified, it will be loaded from there. Otherwise, the dataset will be automatically downloaded to the path specified and used in the run.

After all three clients have been started, federated learning should commence.
