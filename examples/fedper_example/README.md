# FedPer Federated Learning Example
This example provides an example of training a FedPer type model on a non-IID subset of the MNIST data. The FL server
expects three clients to be spun up (i.e. it will wait until three clients report in before starting training). Each client
has a modified version of the MNIST dataset. This modification essentially subsamples a certain number from the original
training and validation sets of MNIST in order to synthetically induce local variations in the statistical properties
of the clients training/validation data. In theory, the models should be able to perform well on their local data
while learning from other clients data that has different statistical properties. The subsampling is specified by
sending a list of integers between 0-9 to the clients when they are run with the argument `--minority_numbers`.

The server has some custom metrics aggregation and uses Federated Averaging as its server-side optimization. The implementation uses a special type of weight exchange based on named-layer identification.

## Running the Example
In order to run the example, first ensure you have [installed the dependencies in your virtual environment according to the main README](/README.md#development-requirements) and it has been activated.

## Starting Server

The next step is to start the server by running
```bash
python -m examples.fedper_example.server  --config_path /path/to/config.yaml
```
from the FL4Health directory. The following arguments must be present in the specified config file:
* `n_clients`: number of clients the server waits for in order to run the FL training
* `local_epochs`: number of epochs each client will train for locally
* `batch_size`: size of the batches each client will train on
* `n_server_rounds`: The number of rounds to run FL
* `downsampling_ratio`: The amount of downsampling to perform for minority digits

## Starting Clients

Once the server has started and logged "FL starting," the next step, in separate terminals, is to start the three
clients. This is done by simply running (remembering to activate your environment)
```bash
python -m examples.fedper_example.client --dataset_path /path/to/data --minority_numbers <sequence of numbers>
```
**NOTE**: The argument `dataset_path` has two functions, depending on whether the dataset exists locally or not. If
the dataset already exists at the path specified, it will be loaded from there. Otherwise, the dataset will be
automatically downloaded to the path specified and used in the run.

The argument `minority_numbers` specifies which digits (0-9) in the MNIST dataset the client will subsample to
simulate non-IID data between clients. For example `--minority_numbers 1 2 3 4 5` will ensure that the client
downsamples these digits (using the `downsampling_ratio` specified to the config).

After the clients have been started federated learning should commence.
