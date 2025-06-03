# Federated Conditional Variational Auto-encoder Example
This is an example implementation of training a Conditional Variational auto-encoder model on the MNIST dataset. In this example, the model is conditioned based on a client-specific ID that is an integer associated with the client. The given client ID is then one hot encoded by having the total number of conditions (in this case clients), to create the condition vector. This allows the model to learn a specific mapping of the input to the latent representation while being conditioned on the specific client feature space, potentially leading to more personalized representations. A data converter (`AutoEncoderDatasetConverter`) is initiated using the condition vector and then used for converting the data into the proper format for training. This converter can be initiated with any user defined vector for the condition, alternatively, user can pass `label` as the condition which conditions each data sample on its label.
In this example, the client is a BasicClient, but it can instead inherit from any of the available client classes. The server uses Federated Averaging to aggregate the CVAE model weights.

## Running the Example
In order to run the example, first ensure you have [installed the dependencies in your virtual environment according to the main README](/README.md#development-requirements) and it has been activated.

## Starting Server

The next step is to start the server by running
```bash
python -m examples.ae_examples.cvae_examples.mlp_cvae_example.server  --config_path /path/to/config.yaml
```
from the FL4Health directory. The following arguments must be present in the specified config file:
* `n_clients`: number of clients the server waits for in order to run the FL training
* `local_epochs`: number of epochs each client will train for locally
* `batch_size`: size of the batches each client will train on
* `n_server_rounds`: The number of rounds to run FL
* `checkpoint_path`: path to save the best server model
* `latent_dim`: size of the latent vector in the CVAE or VAE model

## Starting Clients

Once the server has started and logged "FL starting," the next step, in separate terminals, is to start the two
clients. This is done by simply running (remembering to activate your environment)
```bash
python -m examples.ae_examples.cvae_examples.mlp_cvae_example.client --dataset_path /path/to/data --condition "client's ID number" --num_conditions "total number of clients (client IDs)"
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
python -m examples.ae_examples.cvae_examples.mlp_cvae_example.client --dataset_path examples/datasets/MNIST --condition 0 --num_conditions 2
```
Client 1:
```bash
python -m examples.ae_examples.cvae_examples.mlp_cvae_example.client --dataset_path examples/datasets/MNIST --condition 1 --num_conditions 2
```
