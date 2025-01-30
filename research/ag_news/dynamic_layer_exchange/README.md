# Dynamic Layer Exchange Federated Learning For Fine-tuning BERT
This example leverages the dynamic layer exchanger, where instead of exchanging the entire model between the server and clients in every training round,
only certain tensors are exchanged, and weighted averaging is performed in a per-tensor fashion on the server side.

In each training round, the tensors to be exchanged with the server are selected based their "drift norm" in that training round, which is defined as the l2-norm of the difference between their values at the end of that training round and their initial values at the beginning of the same training round.

You can customize how tensors are selected and how many tensors to exchange, but more on this later.

This example fine-tunes a pre-trained BERT encoder to perform text classification on the AG News dataset in a federated setting.
Each client performs Dirichlet sampling on its training set so we get heterogeneous data distributions across clients.

## Running the Example
In order to run the example, first ensure you have [installed the dependencies in your virtual environment according to the main README](/README.md#development-requirements) and it has been activated.

## Information about Dataset
client_data.py handles all data preprocessing steps and is responsible for constructing the dataloaders for training and evaluation.
Since we are using a pre-trained BERT model, its corresponding tokenizer and vocabulary (all pre-trained) are used for processing
the raw text data. The Dirichlet subsampling on the training data is also performed in this module. The execution of client_data.py is included
as part of the client code, so you do not need to run it separately.

## Starting Server

The next step is to start the server by running something like
```
python -m research.ag_news.dynamic_layer_exchange.server   --config_path /path/to/config.yaml
```
from the FL4Health directory. The following arguments must be present in the specified config file:
* `n_clients`: number of clients the server waits for before starting FL training.
* `local_epochs`: number of epochs each client will train for locally.
* `batch_size`: size of the batches each client will train on.
* `num_classes`: number of classes in the classification task.
* `normalize`: specifies whether division by the tensor's dimension is performed when computing its drift norm.
* `filter_by_percentage`: by using the dynamic layer exchanger, each client takes in an "--exchange_percentage" argument and a "--norm_threshold" argument, each corresponding to a mechanism for selecting the tensors to be exchanged with the server. The argument `filter_by_percentage` toggles between these two mechanisms. See the next section for more details on these two mechanisms, see the next section.
* `sample_percentage`: specifies how much of the original training set is retained after Dirichlet sampling.
* `beta`: a positive real number which controls the heterogeneity of the data distributions across clients. The smaller beta is, the more heterogeneous the distributions are.
* `n_server_rounds`: The number of rounds to run FL.

## Starting Clients

Once the server has started and logged "FL starting," the next step, in separate terminals, is to start the `n_clients`
clients expected by the server. This is done by simply running (remembering to activate your environment)
```
python -m research.ag_news.dynamic_layer_exchange.client --dataset_dir <path_to_dataset> --exchange_percentage exchange_percentage --norm-threshold norm_threshold
```
* `path_to_dataset` is the path towards the directory where the dataset is stored.
* `exchange-percentage` has value $p$, where $p$ is between 0 and 1. If `filter_by_percentage` in the config file is true, then in every training round, the top ceil($p \cdot N$) tensors with the largest (or smallest) drift norms will be exchanged, where $N$ is the total number of tensors.
* `norm-threshold` is a positive real number $t$, and if `filter_by_percentage` in the config file is false, then every tensor with drift norm larger (or smaller) than $t$ will be exchanged.

After `n_clients` clients have been started federated learning should commence.

### Running hyperparameter sweep for this example

To run the hyperparameter sweep you simply run the command

```bash
./research/ag_news/dynamic_layer_exchange/fedadam/run_hp_sweep.sh \
   path_to_config.yaml \
   path_to_folder_for_artifacts/ \
   path_to_folder_for_dataset/ \
   path_to_desired_venv/
```

from the top level directory of the repository

An example is something like
``` bash
./research/ag_news/dynamic_layer_exchange/run_hp_sweep.sh \
   research/ag_news/dynamic_layer_exchange/config.yaml \
   research/ag_news/dynamic_layer_exchange \
   /Users/david/Desktop/ag_news_dataset \
   /h/demerson/vector_repositories/fl4health_env/
```

In order to manipulate the grid search being conducted, you need to change the parameters for `exchange_percentage` and `lr`, the exchange percentages and client learning rates, respectively, in the `run_hp_sweep.sh` script directly.
