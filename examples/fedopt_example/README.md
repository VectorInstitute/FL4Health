# FedOpt Federated Learning Example
This examples expands on the concepts of the first basic example to consider several additional federated learning concepts. Foremost, this implements a server side FedAdam optimization strategy. It also implements several expansions to the metrics aggregation steps. Finally, the example coordinates a significant amount of communication between the server and clients in terms of passing model configurations, trained vocabularies, and label encoders. This is an essential part of the example, as the dataset is an NLP task and the local client datasets are "distributed." Therefore, the server must provide a unified vocabulary and label encoding scheme.

The example also begins to separate out data loader construction and metrics calculations from the client and server code in order to begin abstracting such components.

_NOTE_: This take a fair bit of time to run on CPU

## Running the Example
In order to run the example, first ensure you have [installed the dependencies in your virtual environment according to the main README](/README.md#development-requirements) and it has been activated.

## Dataset Partitioning

Before starting the example, one needs to partition the original news classification dataset, found under `examples/datasets/agnews_data/datasets/AG_NEWS/train.csv` into distinct datasets that will be read by each of the clients. This is done by running
```bash
python -m examples.datasets.partition_dataset --original_dataset_path /path/to/ag_news/train.csv
--partition_dir /path/to/partition/destination/
--partitioner_config_path examples/datasets/agnews_data/partitioner_config.json
--n_partitions 3
--overwrite
```
These arguments specify the following
* `original_dataset_path`: Path to the ag_news train.csv dataset. This is, by default, housed in the directory specified in the beginning of this section
* `partition_dir`: Path where the dataset partitions should go
* `partitioner_config_path`: Path to the config file for the partitioner. The default is housed in `examples/datasets/agnews_data/datasets/AG_NEWS/train.csv`
* `n_partitions`: Number of distinct partitions to break the original dataset into
* `overwrite`: Specifies whether to overwrite the directory specified. If the directory already exists and overwrite is not specified, the partition script with throw an error.

## Starting Server

The next step is to start the server by running something like
```bash
python -m examples.fedopt_example.server   --config_path /path/to/config.yaml
```
from the FL4Health directory. The following arguments must be present in the specified config file:
* `n_clients`: number of clients the server waits for in order to run the FL training
* `local_epochs`: number of epochs each client will train for locally
* `batch_size`: size of the batches each client will train on
* `vocab_dimension`: embedding dimension of the word embeddings
* `hidden_size`: hidden size of the LSTM layers
* `sequence_length`: input sequence length of the LSTM model to be trained
* `n_server_rounds`: The number of rounds to run FL

## Starting Clients

Once the server has started and logged "FL starting," the next step, in separate terminals, is to start the `n_clients`
clients expected by the server. This is done by simply running (remembering to activate your environment)
```bash
python -m examples.fedopt_example.client --dataset_path <path_to_local_dataset>
```
* `path_to_local_dataset` should correspond to the partition destination provide in running the partition_dataset script.
After `n_clients` clients have been started federated learning should commence.
