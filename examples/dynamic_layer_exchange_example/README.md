# Dynamic Layer Exchange Federated Learning Example
This example demonstrates the usage of the dynamic layer exchanger. At the end of each training round, each client selects an arbitrary set of its layers to be exchanged with the server.

On the server side, the same layers are grouped together, then weighted averaging is performed in a per-layer fashion.

In this example, the dataset is an augmented version of the CIFAR-10 dataset that is non--IID. The FL server expects three clients to be spun up (i.e. it will wait until three clients report in before starting training). Each client has a modified version of the CIFAR-10 dataset. This modification essentially subsamples a certain number from the original training and validation sets of CIFAR-10 in order to synthetically induce local variations in the statistical properties of the clients training/validation data. In theory, the models should be able to perform well on their local data while learning from other clients data that has different statistical properties. The proportion of labels at each client is determined by Dirichlet distribution across the classes.

## Running the Example
In order to run the example, first ensure you have [installed the dependencies in your virtual environment according to the main README](/README.md#development-requirements) and it has been activated.

## Starting Server

The next step is to start the server by running
```bash
python -m examples.dynamic_layer_exchange_example.server  --config_path /path/to/config.yaml
```
from the FL4Health directory. The following arguments must be present in the specified config file:
* `n_clients`: number of clients the server waits for in order to run the FL training
* `local_epochs`: number of epochs each client will train for locally
* `batch_size`: size of the batches each client will train on
* `n_server_rounds`: The number of rounds to run FL
* `normalize`: specifies whether division by the tensor's dimension is performed when computing its drift norm.
* `filter_by_percentage`: by using the dynamic layer exchanger, each client takes in an "--exchange_percentage" argument and a "--norm_threshold" argument, each corresponding to a mechanism for selecting the tensors to be exchanged with the server. The argument `filter_by_percentage` toggles between these two mechanisms. See the next section for more details on these two mechanisms, see the next section.
* `exchange_percentage`: a real number $p$ between 0 and 1. If `filter_by_percentage` is true, then in every training round, the top ceil($p \cdot N$) tensors with the largest (or smallest) drift norms will be exchanged, where $N$ is the total number of tensors.
* `norm_threshold`: a positive real number $t$. If `filter_by_percentage` in the config file is false, then every layer with drift norm larger (or smaller) than $t$ will be exchanged.
* `sample_percentage`: specifies how much of the original training set is retained after Dirichlet sampling.
* `beta`: a positive real number which controls the heterogeneity of the data distributions across clients. The smaller beta is, the more heterogeneous the distributions are.

## Starting Clients

Once the server has started and logged "FL starting," the next step, in separate terminals, is to start the four
clients. This is done by simply running (remembering to activate your environment)
```bash
python -m examples.dynamic_layer_exchange_example.client --dataset_path /path/to/data
```
**NOTE**: The argument `dataset_path` has two functions, depending on whether the dataset exists locally or not. If
the dataset already exists at the path specified, it will be loaded from there. Otherwise, the dataset will be
automatically downloaded to the path specified and used in the run.

After all 3 clients have been started federated learning should commence.
