# Warmed Up FedProx Federated Learning Example

In this example, FedProx is applied to a modified version of the MNIST dataset that is non--IID after being warmed up with FedAvg. The FL server expects three clients to be spun up (i.e. it will wait until three clients report in before starting training). Each client has it's own modified version of the MNIST dataset. This modification essentially subsamples a certain number from the original training and validation sets of MNIST in order to synthetically induce local variations in the statistical properties of the clients training/validation data. In theory, the models should be able to perform well on their local data while learning from other clients data that has different statistical properties. The proportion of labels at each client is determined by Dirichlet distribution across the classes. The lower the beta parameter is for each class, the higher the degree of the label heterogeneity.

The server has some custom metrics aggregation and uses FedProx as its server-side optimization.

After the warm-up training, clients can load their warmed-up models and continue training with the FedProx algorithm. To maintain consistency in the data loader between both runs, it is crucial to set a fixed seed for both clients and the server, ensuring uniformity in random data points across consecutive runs. Therefore, we ensure a fixed seed is set for these consecutive runs in both the `client.py` and `server.py` files. Additionally, to load the warmed-up models, it's important provide the path to the pretrained models based on client's unique name, ensuring that we can load the trained local model for each client from the previous example as a warmed-up model. Since models in the two runs can be different, loading weights from the pretrained model requires providing a mapping between the pretrained model and the model used in FL training. This mapping is accomplished through the `weights_mapping.json` file, which contains the names of the pretrained model's layers and the corresponding names of the layers in the model used in FL training.

### Running the Example

In order to run the example, first ensure you have [installed the dependencies in your virtual environment according to the main README](/README.md#development-requirements) and it has been activated.

## Starting Server

The next step is to start the server by running
```bash
python -m examples.warm_up_example.warmed_up_fedprox.server --config_path /path/to/config.yaml --seed "SEED"
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
python -m examples.warm_up_example.warmed_up_fedprox.client --dataset_path /path/to/data --seed "SEED" --pretrained_model_path /path/to/model_checkpoint  --weights_mapping_path /path/to/weights/mapping/file
```
**NOTE**: The argument `dataset_path` has two functions, depending on whether the dataset exists locally or not. If the dataset already exists at the path specified, it will be loaded from there. Otherwise, the dataset will be automatically downloaded to the path specified and used in the run.

**NOTE**: "SEED" above should match that of the warm up run if you want to ensure the datasets are split consistently.

After all three clients have been started, federated learning should commence.
