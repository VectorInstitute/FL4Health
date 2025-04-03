# Federated Convolutional CVAE Example
This is an example implementation of training a Convolutional Conditional Variational Auto-encoder model on the MNIST dataset. In this example, each data sample has a specific condition that is the binary representation of their class. To do so, a data converter (`AutoEncoderDatasetConverter`) is initiated with a customized converter function (`binary_class_condition_data_converter`), then used for converting the data into the proper format for training. Alternatively, user can pass `label` as the condition with `do_one_hot_encoding= True` which conditions each data sample on its one hot encoded vector of the label. Notably, all the clients should use the same converter function for consistency. In the encoder module, the condition is fed after the convolutional layers. Even though the client in this example is a BasicClient, users have the option to inherit from any of the available client classes instead. The server uses Federated Averaging to aggregate the CVAE model weights.


## Running the Example
In order to run the example, first ensure you have [installed the dependencies in your virtual environment according to the main README](/README.md#development-requirements) and it has been activated.

## Starting Server

The next step is to start the server by running
```bash
python -m examples.ae_examples.cvae_examples.conv_cvae_example.server  --config_path /path/to/config.yaml
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
python -m examples.cvae_example.conv_cvae_example.client --dataset_path /path/to/data
```
**NOTE**: The argument `dataset_path` has two functions, depending on whether the dataset exists locally or not. If
the dataset already exists at the path specified, it will be loaded from there. Otherwise, the dataset will be
automatically downloaded to the path specified and used in the run.

After both clients have been started federated learning should commence.


### In this example
You can use the following commands to run the clients.

```bash
python -m examples.ae_examples.cvae_examples.conv_cvae_example.client --dataset_path examples/datasets/MNIST
```
