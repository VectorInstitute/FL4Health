# FedDG-GA Example
This is an example of [Federated Domain Generalization with Generalization Adjustment](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Federated_Domain_Generalization_With_Generalization_Adjustment_CVPR_2023_paper.pdf)
(FedDG-GA). FedDG-GA is a [Domain Generalization](https://paperswithcode.com/task/domain-generalization) method
that aims to produce a model that is better generalizable by implementing a variance reduction regularizer.
It aims to achieve a tighter generalization with an explicit re-weighted aggregation as opposed to the implicit
multi-domain data-sharing of conventional domain generalization methods.

FedDG-GA is also a method that can be combined with other FL algorithms to improve their results. In this demo,
FedDG-GA is applied in conjunction with APFL to an augmented version of the MNIST dataset that is non--IID. The FL
server expects two clients to be spun up (i.e. it will wait until two clients report in before starting training). Each
client has a modified version of the MNIST dataset. This modification essentially subsamples a certain number from the
original training and validation sets of MNIST in order to synthetically induce local variations in the statistical
properties  of the clients training/validation data. The proportion of labels at each client is determined by Dirichlet
distribution across the classes. The lower the beta parameter is for each class, the higher the degree of the label
heterogeneity.

## Running the Example
In order to run the example, first ensure you have [installed the dependencies in your virtual environment according to the main README](/README.md#development-requirements) and it has been activated.

## Starting Server

The next step is to start the server by running
```bash
python -m examples.feddg_ga_example.server  --config_path /path/to/config.yaml
```
from the FL4Health directory. The following arguments must be present in the specified config file:
* `n_clients`: number of clients the server waits for in order to run the FL training
* `local_epochs`: number of epochs each client will train for locally
* `batch_size`: size of the batches each client will train on
* `n_server_rounds`: The number of rounds to run FL
* `evaluate_after_fit`: Should be set to `True`. Performs an evaluation at the end of each client's fit round.
* `pack_losses_with_val_metrics`: Should be set to `True`. Includes validation losses with metrics calculations

## Starting Clients

Once the server has started and logged "FL starting," the next step, in separate terminals, is to start the two
clients. This is done by simply running (remembering to activate your environment)
```bash
python -m examples.feddg_ga_example.client --dataset_path /path/to/data
```
**NOTE**: The argument `dataset_path` has two functions, depending on whether the dataset exists locally or not. If
the dataset already exists at the path specified, it will be loaded from there. Otherwise, the dataset will be
automatically downloaded to the path specified and used in the run.

After all clients have been started, federated learning should commence.
