# Federated Dimensionality reduction using CVAEs Example
In this example, clients first reduce the dimensionality of the data using an already federally trained Conditional Variational Auto-encoder (please refer to `cvae_examples/mlp_cvae_example`). Then, the data in the reduced dimension is used to train a small neural network model (`examples/models/mnist_model.py`). Federated training for the main task of MNIST classification can be done with any of the FL approaches by specifying the parent client class. The default parent client class in this example is the `BasicClient` and the server uses Federated Averaging.

## Running the Example
In order to run the example, first ensure you have [installed the dependencies in your virtual environment according to the main README](/README.md#development-requirements) and it has been activated.

## Starting Server

The next step is to start the server by running
```bash
python -m examples.ae_examples.cvae_dim_example.server  --config_path /path/to/config.yaml
```
from the FL4Health directory. The following arguments must be present in the specified config file:
* `n_clients`: number of clients the server waits for in order to run the FL training
* `local_epochs`: number of epochs each client will train for locally
* `batch_size`: size of the batches each client will train on
* `n_server_rounds`: The number of rounds to run FL
* `checkpoint_path`: path to save the best server model
* `latent_dim`: size of the latent vector in the CVAE or VAE model
* `cvae_model_path`: path to the saved CVAE model for dimensionality reduction

**NOTE**: Instead of using a global CVAE for all the clients, you can pass personalized CVAE models to each client, but make sure that these models are previously trained in an FL setting, and are not very different, otherwise, that can lead the dimensionality reduction to map the data samples into different latent spaces which might increase the heterogeneity.

## Starting Clients

Once the server has started and logged "FL starting," the next step, in separate terminals, is to start the two
clients. This is done by simply running (remembering to activate your environment)
```bash
python -m examples.ae_examples.cvae_dim_example.client --dataset_path /path/to/data --condition "client's ID number" --num_conditions "total number of clients (client IDs)"
```
**NOTE**: The argument `dataset_path` has two functions, depending on whether the dataset exists locally or not. If
the dataset already exists at the path specified, it will be loaded from there. Otherwise, the dataset will be
automatically downloaded to the path specified and used in the run.

After both clients have been started federated learning should commence.

**NOTE**: In this example, the argument `condition ` is used to set a client-specific condition on the CVAE model. Here, client IDs are used as the condition on their data. The next argument `num_conditions` is used to create a one hot encoded condition vector for each client.


### In this example
You can use the following commands to run the clients.
Client 0:
```bash
python -m examples.ae_examples.cvae_dim_example.client --dataset_path examples/datasets/MNIST --condition 0 --num_conditions 2
```
Client 1:
```bash
python -m examples.ae_examples.cvae_dim_example.client --dataset_path examples/datasets/MNIST --condition 1 --num_conditions 2
```
