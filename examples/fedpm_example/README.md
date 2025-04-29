# FedPM Federated Learning Example
This is an example of Federated Probabilistic Mask Training (FedPM), as described in the paper "[Sparse Random Networks for Communication-efficient Federated Learning](http://arxiv.org/pdf/2209.15328)". In this regime, each client randomly initializes a network and learns how to prune that random network to find a sub-network which performs the given task well. This is achieved by training Bernoulli probability scores associated with the model parameters while keeping the model parameters themselves fixed.

In this particular example, the FL server
expects three clients to be spun up (i.e. it will wait until three clients report in before starting training). Each
client has a modified version of the MNIST dataset. This modification essentially subsamples a certain number from the
original training and validation sets of MNIST in order to synthetically induce local variations in the statistical
properties of the clients training/validation data. In theory, the models should be able to perform well on their
local data while learning from other clients data that has different statistical properties. The subsampling is
specified by sending a list of integers between 0-9 to the clients when they are run with the argument
`--minority_numbers`.
The model in this example, which is a CNN with 4 convolutional layers followed by 3 fully connected layers, is selected from the original paper.

On the client side, local training only updates the probability scores, and in every FL round, after local training has concluded, each client uses its probability scores to perform Bernoulli sampling and produces a binary mask for each of its tensors. These masks are then sent to the server for aggregation.

On the server side, based on the conjugate relation between the Beta and Bernoulli distributions, Bayesian aggregation is performed to aggregate the binary masks and update the Bernoulli probability scores. Given any fitting round, the priors (i.e., the parameters of the Beta distribution) for the next round is determined by the aggregation outcome of the current round and reset to all ones every couple of rounds. The user may specify how frequent these priors should be reset via the `priors_reset_frequency` parameter in the config file.

## Running the Example
In order to run the example, first ensure you have installed the dependencies in your virtual environment
[according to the main README](/README.md#development-requirements) and it has been activated.

## Starting Server

The next step is to start the server by running
```bash
python -m examples.fedpm_example.server  --config_path /path/to/config.yaml
```
from the FL4Health directory. The following arguments must be present in the specified config file:
* `n_clients`: number of clients the server waits for in order to run the FL training
* `local_epochs`: number of epochs each client will train for locally
* `batch_size`: size of the batches each client will train on
* `n_server_rounds`: the number of rounds to run FL
* `downsampling_ratio`: the amount of downsampling to perform for minority digits
* `priors_reset_frequency`: the frequency with which the priors of the Beta distribution involved in the Bayesian aggregation process is reset. If `priors_reset_frequency` is `n`, then the priors are reset every `n` fitting rounds.

## Starting Clients

Once the server has started and logged "FL starting," the next step, in separate terminals, is to start the three
clients. This is done by simply running (remembering to activate your environment)
```bash
python -m examples.fedpm_example.client --dataset_path /path/to/data --minority_numbers <sequence of numbers>
```
**NOTE**: The argument `dataset_path` has two functions, depending on whether the dataset exists locally or not. If
the dataset already exists at the path specified, it will be loaded from there. Otherwise, the dataset will be
automatically downloaded to the path specified and used in the run.

The argument `minority_numbers` specifies which digits (0-9) in the MNIST dataset the client will subsample to
simulate non-IID data between clients. For example `--minority_numbers 1 2 3 4 5` will ensure that the client
downsamples these digits (using the `downsampling_ratio` specified to the config).

After the clients have been started federated learning should commence.
