# Federated Learning with Local Fine-Tuning

This example is a very simple extension of the basic federated learning example to perform "local fine-tuning" for client models. This is a very basic way of providing model "personalization" for clients. The idea is that client models are trained with a federated learning approach, which can be almost any way of doing federated learning in general, such as, non-personalized approaches like FedAvg. These clients learn global models using each others data for some number of server rounds. Thereafter, the models are each fine-tuned exclusively on a clients own data. In this way, each client has a distinct model that should perform better on its own data. For additional documentation about the setup for the other components of this example. See the "basic_example" README.

It is important to note that local fine-tuning has a number of weaknesses. The first is that any model learned in this way may not generalize outside of the clients own data very well. As it has been fine-tuned strictly on that data domain. Further, without special treatment, in the presence of strong data heterogeneity, this approach will suffer from the same pitfalls that FedAvg does and the fine-tuning maybe be starting from a poor global model.

## Running the Example

In order to run the example, first ensure you have [installed the dependencies in your virtual environment according to the main README](/README.md#development-requirements) and it has been activated.

## Starting Server

The next step is to start the server by running
```bash
python -m examples.fl_plus_local_ft_example.server  --config_path /path/to/config.yaml
```
from the FL4Health directory. The following arguments must be present in the specified config file:
* `n_clients`: number of clients the server waits for in order to run the FL training
* `local_epochs`: number of epochs each client will train for locally
* `batch_size`: size of the batches each client will train on
* `n_server_rounds`: The number of rounds to run FL

## Starting Clients

Once the server has started and logged "FL starting," the next step, in separate terminals, is to start the two
clients. This is done by simply running (remembering to activate your environment)
```bash
python -m examples.fl_plus_local_ft_example.client --dataset_path /path/to/data
```
**NOTE**: The argument `dataset_path` has two functions, depending on whether the dataset exists locally or not. If
the dataset already exists at the path specified, it will be loaded from there. Otherwise, the dataset will be
automatically downloaded to the path specified and used in the run.

After both clients have been started federated learning should commence.
