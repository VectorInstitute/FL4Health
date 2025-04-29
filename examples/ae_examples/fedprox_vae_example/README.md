# FedProx Federated Variational Auto-encoder Example
This is an example implementation of training a Variational Auto-encoder model on the MNIST dataset. This example uses the FedProx strategy to train personalized variational autoencoders with adaptive proximal weight. The server uses Federated Averaging to aggregate the VAE model weights. The structure of the encoder and decoder models are defined in `models.py``.

## Running the Example
In order to run the example, first ensure you have [installed the dependencies in your virtual environment according to the main README](/README.md#development-requirements) and it has been activated.

## Starting Server

The next step is to start the server by running
```bash
python -m examples.ae_examples.fedprox_vae_example.server  --config_path /path/to/config.yaml
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
python -m examples.ae_examples.fedprox_vae_example.client --dataset_path /path/to/data
```
**NOTE**: The argument `dataset_path` has two functions, depending on whether the dataset exists locally or not. If
the dataset already exists at the path specified, it will be loaded from there. Otherwise, the dataset will be
automatically downloaded to the path specified and used in the run.

After both clients have been started federated learning should commence.

### In this example
You can use the following commands to run the clients.
```bash
python -m examples.ae_examples.fedprox_vae_example.client --dataset_path examples/datasets/MNIST
```
