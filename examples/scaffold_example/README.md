# SCAFFOLD Federated Learning Example
This is an example of [Stochastic Controlled Averaging for Federated Learning](https://arxiv.org/pdf/1910.06378.pdf)(SCAFFOLD). SCAFFOLD is a popular method for federated learning in situations where data across clients is heterogenous (non-iid). In these cases, FedAvg suffers from client drift resulting in unstable and slow
convergence. To surmount this, SCAFFOLD uses control variates to correct for client drift during local updates. This is shown to decrease the number of communication rounds required when compared to other approaches to Federated Learning such as FedAvg.

In this demo, SCAFFOLD is applied to an augmented version of the MNIST dataset that is non--IID. The FL server expects three clients to be spun up (i.e. it will wait until three clients report in before starting training). Each client has a modified version of the MNIST dataset. This modification essentially subsamples a certain number from the original training and validation sets of MNIST in order to synthetically induce local variations in the statistical properties of the clients training/validation data. In theory, the models should be able to perform well on their local data while learning from other clients data that has different statistical properties. The proportion of labels at each client is determined by Dirichlet distribution across the classes. The lower the beta parameter is for each class, the higher the degree of the label heterogeneity.

# Running the Example
In order to run the example, first ensure you have [installed the dependencies in your virtual environment according to the main README](/README.md#development-requirements) and it has been activated.

## Starting Server

The next step is to start the server by running
```bash
python -m examples.scaffold_example.server  --config_path /path/to/config.yaml
```
from the FL4Health directory. The following arguments must be present in the specified config file:
* `n_clients`: number of clients the server waits for in order to run the FL training
* `local_steps`: number of steps (1 per batch) each client will train for locally
* `batch_size`: size of the batches each client will train on
* `n_server_rounds`: The number of rounds to run FL

## Starting Clients

Once the server has started and logged "FL starting," the next step, in separate terminals, is to start the three
clients. This is done by simply running (remembering to activate your environment)
```bash
python -m examples.scaffold_example.client --dataset_path /path/to/data
```
**NOTE**: The argument `dataset_path` has two functions, depending on whether the dataset exists locally or not. If
the dataset already exists at the path specified, it will be loaded from there. Otherwise, the dataset will be
automatically downloaded to the path specified and used in the run.

After all three clients have been started, federated learning should commence.
