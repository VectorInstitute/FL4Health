
This example provides a very simple implementation of federated self supervised learning (SSL) with FedSimCLR on the CIFAR dataset.
FedSimCLR is a straightforward generalization of SimCLR (https://arxiv.org/pdf/2002.05709) to the federated setting presented in
the Fed-X paper (https://arxiv.org/pdf/2207.09158). FL server expects two clients to be spun up (i.e. it will wait until two
clients report in before starting training). Each client has the same "local" dataset. I.e. they each load the complete CIFAR dataset
and therefore have the same training and validation sets. The underlying dataset that is used (and must be used for any SSL) is an
SslTensorDataset which loads an image and its corresponding transformed version. Following SimCLR and FedSimCLR, a contrastive loss
(NT-Xent https://proceedings.neurips.cc/paper_files/paper/2016/file/6b180037abbebea991d8b1232f8a8ca9-Paper.pdf) is used as the objective.
In order to ensure both the input and the target are mapped to features by the model, we extend the client with a simple method
`transform_target` that takes in a target (in this case a transformed version of image) and obtains its features representation.
The server has some custom metrics aggregation, but is otherwise a vanilla FL implementation using FedAvg as the server side optimization.
The server checkpoints the model that obtains the lowest average performance on the client validation sets. This model will be used in the
follow up example at `examples/fedsimclr_example/fedsimclr_finetuning_example` as the basis for finetuning.

## Running the Example
In order to run the example, first ensure you have [installed the dependencies in your virtual environment according to the main README](/README.md#development-requirements) and it has been activated.

## Starting Server

The next step is to start the server by running
```bash
python -m examples.fedsimclr_example.fedsimclr_pretraining_example.server  --config_path /examples/fedsimclr_example/fedsimclr_pretraining_example/config.yaml
```
from the FL4Health directory. The following arguments must be present in the specified config file:
* `n_clients`: number of clients the server waits for in order to run the FL training
* `local_epochs`: number of epochs each client will train for locally
* `batch_size`: size of the batches each client will train on
* `n_server_rounds`: The number of rounds to run FL
* `checkpoint_path`: The path to checkpoint the best performing model.

## Starting Clients

Once the server has started and logged "FL starting," the next step, in separate terminals, is to start the two
clients. This is done by simply running (remembering to activate your environment)
```bash
python -m examples.fedsimclr_example.fedsimclr_pretraining_example.client  --dataset_path /path/to/data/
```
**NOTE**: The argument `dataset_path` has two functions, depending on whether the dataset exists locally or not. If
the dataset already exists at the path specified, it will be loaded from there. Otherwise, the dataset will be
automatically downloaded to the path specified and used in the run.

After both clients have been started federated learning should commence.
