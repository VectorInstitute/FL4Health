# BERT fine-tuning example
In this example, a pre-trained BERT encoder is fine-tuned on the AG News classification task using the FedAvg strategy. The AG News dataset is a collection of more than 1 million news articles classified into four categories: “World”, “Sports”, “Business”, and “Sci/Tech”. Each client performs Dirichlet sampling on the AG News training set, resulting in heterogeneous data distributions across clients. Server-side checkpointing is used to save the latest as well as the best server model.
This example is a simplified version of the research code in `research/ag_news`, where partial parameter exchange techniques are explored.

## Dataset pre-processing
`client_data.py` handles all data preprocessing steps and is responsible for constructing the dataloaders for training and evaluation.
Since we are using a pre-trained BERT model, its corresponding tokenizer and vocabulary (all pre-trained) are used for processing
the raw text data. The Dirichlet subsampling on the training data is also performed in this example. `sample_percentage` in `config` is the downsampling of the entire data that is assigned to each client. `beta` controls the heterogeneity of data distribution among clients; smaller beta values make the clients' distributions more non-IID.
The execution of `client_data.py` is included
as part of the client code, so you do not need to run it separately.

## Running the example
To run the example, first ensure you have [installed the dependencies in your virtual environment according to the main README](/README.md#development-requirements) and it has been activated.

## Starting Server

The next step is to start the server by running
```bash
python -m examples.bert_finetuning_example.server --config_path /path/to/config.yaml
```
from the FL4Health directory. The following arguments must be present in the specified config file:
* `n_clients`: number of clients the server waits for in order to run the FL training
* `local_epochs`: number of epochs each client will train for locally
* `batch_size`: size of the batches each client will train on
* `n_server_rounds`: the number of rounds to run FL
* `num_classes`: number of classes in the classification task
* `checkpoint_path`: path to save the server's checkpoints
* `sample_percentage`: the proportion of the training data that will be sampled for each client. Should be between 0 and 1 inclusive.
* `beta`: a positive real number that controls the heterogeneity of the data distributions among clients. The smaller beta is, the more heterogeneous the distributions are.

## Starting Clients

Once the server has started and logged "FL starting," the next step, in separate terminals, is to start the two
clients. This is done by simply running (remembering to activate your environment)
```bash
python -m examples.bert_finetuning_example.client
```
**NOTE**: The dataset is loaded explicitly from Hugging Face and therefore the `data_path` is not specified.

After both clients have been started, federated learning should commence.

## Run on Cluster
The `run.sh` script runs the server and two clients. For more clients, increase the `NUM_CLIENTS` value in the script. For more clients or larger batch sizes, you might need to use an `a40` GPU and request larger memory.

To launch the server and clients, run:

```bash
sbatch examples/bert_finetuning_example/run.sh \
  path_to_config.yaml \
  path_to_folder_for_artifacts/ \
  path_to_folder_for_dataset/ \
  path_to_desired_venv/ \
  client_learning_rate_value \
  server_address
```
