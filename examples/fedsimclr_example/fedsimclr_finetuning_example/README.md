# Federated Finetuning of Self Supervised Model.
This example provides a very simple implementation finetuning a FedSimCLR model. For information about the pretraining stage,
that needs to be run prior to this, please refer to [fedsimclr_pretraining_example](/examples/fedsimclr_example/fedsimclr_pretraining_example). Assuming pretraining has occurred, a checkpoint to the best performing model on the validation set will be saved. This script will load the saved model, swap the projection head for a prediction head and finetune the model on a small subset of examples. Since the pretraining script uses the training set, the finetuning script uses the test set which is split into training (80%) and validation (20%).


## Running the Example
In order to run the example, first ensure you have [installed the dependencies in your virtual environment according to the main README](/README.md#development-requirements) and it has been activated.

## Starting Server

The next step is to start the server by running
```bash
python -m examples.fedsimclr_example.fedsimclr_finetuning_example.server  --config_path /path/to/config.yaml
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
python -m examples.fedsimclr_example.fedsimclr_finetuning_example.client --dataset_path /path/to/data
```
**NOTE**: The argument `dataset_path` has two functions, depending on whether the dataset exists locally or not. If
the dataset already exists at the path specified, it will be loaded from there. Otherwise, the dataset will be
automatically downloaded to the path specified and used in the run.

After both clients have been started federated learning should commence.
