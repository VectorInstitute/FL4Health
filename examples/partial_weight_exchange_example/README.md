# Partial Weight Exchange Federated Learning Example
This example leverages the partial weight exchange system, where instead of exchanging the entire model between the server and clients in every training round,
only certain parameters are exchanged, and weighted averaging is performed in a per-parameter fashion on the server side.

In each training round, the parameters to be sent for exchange are selected based their "drift norm" in that training round, which is defined as the l2-norm of the difference between their values at the end of that training round and their initial values at the beginning of the same training round. More details about the algorithm and some experiment results using this example are included in "Partial_Weight_Exchange_in_Federated_Learning.pdf".

You can customize how parameters are selected and how many parameters to exchange, but more on this later.

This example uses a pre-trained RoBERTa-base encoder to perform text classification on the AG News dataset in a federated setting, where each client performs a Dirichlet
sampling on its training set so we get heterogeneous data distributions across clients.

## Running the Example
In order to run the example, first ensure you have the virtual env of your choice activated and run
```
pip install --upgrade pip
pip install -r requirements.txt
```
to install all of the dependencies for this project.

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
* `sequence_length`: input sequence length to Roberta Encoder, must be at least 256 and at most 514.
* `normalize`: specifies whether division by the parameter's dimension is performed when computing its drift norm.
* `filter_by_percentage`: each client takes in an "--exchange-percentage" argument and a "--norm-threshold" argument, where each one corresponds to a mechanism for selecting the parameters to be exchanged between the server and clients. The argument `filter_by_percentage` toggles between these two modes.

More precisely, "--exchange-percentage" has value $p$, where $p$ is between 0 and 1. If `filter_by_percentage` is true, then in every training round, the top ceil($p &sdot N$) parametes with the largest drift norms will be exchanged, where $N$ is the total number of parameters.

Alternatively, "--norm-threshold" is a positive real number $t$, and if `filter_by_percentage` is false, then every parameter with drift norm larger than $t$ will be exchanged.

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
