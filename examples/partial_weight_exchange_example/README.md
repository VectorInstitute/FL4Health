# Partial Weight Exchange Federated Learning Example
This example leverages the dynamic layer exchanger, where instead of exchanging the entire model between the server and clients in every training round,
only certain tensors are exchanged, and weighted averaging is performed in a per-tensor fashion on the server side.

In each training round, the tensors to be exchanged with the server are selected based their "drift norm" in that training round, which is defined as the l2-norm of the difference between their values at the end of that training round and their initial values at the beginning of the same training round.

You can customize how tensors are selected and how many tensors to exchange, but more on this later.

This example fine-tunes a pre-trained RoBERTa-base encoder to perform text classification on the AG News dataset in a federated setting, where each client performs a Dirichlet
sampling on its training set so we get heterogeneous data distributions across clients.

## Running the Example
In order to run the example, first ensure you have [installed the dependencies in your virtual environment according to the main README](/README.md#development-requirements) and it has been activated.

## Information about Dataset
client_data.py handles all data preprosessiong steps and is responsible for constructing the dataloaders for training and evaluation.
Since we used the pre-trained RoBERTa-base encoder, its corresponding tokenizer and vocabulary (all pre-trained) are used for processing
the raw text data. The Dirichlet subsampling on the training data is also performed in this module. The execution of client_data.py is included
as part of the client code, so you do not need to run it separately.

## Starting Server

The next step is to start the server by running something like
```
python -m examples.partial_weight_exchange_example.server   --config_path /path/to/config.yaml
```
from the FL4Health directory. The following arguments must be present in the specified config file:
* `n_clients`: number of clients the server waits for in order to run the FL training.
* `local_epochs`: number of epochs each client will train for locally.
* `batch_size`: size of the batches each client will train on.
* `num_classes`: number of classes in the classification task.
* `sequence_length`: input sequence length to RoBERTa Encoder, must be at least 256 and at most 512.
* `normalize`: specifies whether division by the tensor's dimension is performed when computing its drift norm.
* `filter_by_percentage`: each client takes in an "--exchange_percentage" argument and a "--norm_threshold" argument, where each one corresponds to a mechanism for selecting the tensors to be exchanged between the server and clients. The argument `filter_by_percentage` toggles between these two modes.

More precisely, "--exchange-percentage" has value $p$, where $p$ is between 0 and 1. If `filter_by_percentage` is true, then in every training round, the top ceil($p \cdot N$) tensors with the largest drift norms will be exchanged, where $N$ is the total number of tensors.

Alternatively, "--norm-threshold" is a positive real number $t$, and if `filter_by_percentage` is false, then every tensor with drift norm larger than $t$ will be exchanged.

* `sample_percentage`: specifies how much of the original training set is retained after Dirichlet sampling.
* `beta`: a positive real number which controls the heterogeneity of the data distributions across clients. The larger beta is, the more heterogeneous the distributions are.
* `n_server_rounds`: The number of rounds to run FL.

## Starting Clients

Once the server has started and logged "FL starting," the next step, in separate terminals, is to start the `n_clients`
clients expected by the server. This is done by simply running (remembering to activate your environment)
```
python -m examples.partial_weight_exchange_example.client --dataset_path <path_to_dataset> --exchange_percentage exchange_percentage --threshold-value threshold_value
```
* `path_to_dataset` is the path towards the directory where the dataset is stored.
After `n_clients` clients have been started federated learning should commence.

### Running this example on Vector's clusters
The slurm and bash scripts `run_client.slrm`, `run_server.slrm`, and `run_fl_cluster.sh` are used to run this example on Vector's clusters.

* `run_server.slrm` is responsible for starting the server.
* `run_client.slrm` is responsible for starting a client based on the server's address.
* `run_fl_cluster.sh` is responsible for conducting Federated Learning by leveraging the previous two scripts. This is the only script you need to run.

An example usage would be running a command (from the FL4Health directory) such as

```
sh
./examples/partial_weight_exchange_example/run_fl_cluster.sh 8082 examples/partial_weight_exchange_example/config.yaml examples/partial_weight_exchange_example/server_logs examples/partial_weight_exchange_example/client_logs /h/yuchongz/flenv/ 5 0.4
```

where:

* 8082 is the server's address.
* examples/partial_weight_exchange_example/config.yaml is the path to the config file.
* examples/partial_weight_exchange_example/server_logs is the path to the server's logs.
* examples/partial_weight_exchange_example/client_logs is the "base path" to the clients' logs. The script appends "_0.4" to the end of this base path produce the actual path to the logs of the clients.
* /h/yuchongz/flenv/ is the path to the virtual environment.
* 5 is the number of clients.
* 0.4 is the exchange percentage.

Note: the exchange percentage here is only used for creating the path to the clients' logs. The actual exchange percentage used in the experiments is specified in `config.yaml`. You should ensure they are the same.
