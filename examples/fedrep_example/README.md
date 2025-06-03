# FedRep Federated Learning Example
This example provides an example of training a sequentially split type model using [FedRep](https://arxiv.org/pdf/2102.07078.pdf) on a non-IID subset of the CIFAR data. The FL server expects three clients to be spun up (i.e. it will wait until three clients report in before starting training).  Each client has a modified version of the CIFAR dataset. This modification essentially subsamples a certain number from  the original training and validation sets of CIFAR in order to synthetically induce local variations in the statistical  properties of the clients training/validation data. In theory, the models should be able to perform well on their local data while learning from other clients data that has different statistical properties. The proportion of labels at each client  is determined by Dirichlet distribution across the classes. The properties of the Dirichlet distribution modifications are determined by the `sample_percentage` and `beta` values specified in the config. For additional details see the documentation of `DirichletLabelBasedSampler`

The server has some custom metrics aggregation and uses Federated Averaging as its server-side optimization. The implementation here uses a fixed layer exchanger such that only a portion of the model is exchanged with the server for aggregation. In FedRep, only the representation or feature extraction sub module of the model is aggregated on the server side.

## Running the Example
In order to run the example, first ensure you have [installed the dependencies in your virtual environment according to the main README](/README.md#development-requirements) and it has been activated.

## Starting Server

The next step is to start the server by running
```bash
python -m examples.fedrep_example.server  --config_path /path/to/config.yaml
```
from the FL4Health directory. The following arguments must be present in the specified config file:
* `n_clients`: number of clients the server waits for in order to run the FL training
* `local_head_steps`: number of steps each client will train the models **head** module for locally
* `local_rep_steps` : number of steps each client will train the models **representation** module for locally
* `batch_size`: size of the batches each client will train on. This is fixed to be equal for training of the head and representation models, respectively. Different batch sizes are currently **not** supported.
* `n_server_rounds`: The number of rounds to run FL
* `sample_percentage`: specifies how much of the original training set is retained after Dirichlet sampling.
* `beta`: a positive real number which controls the heterogeneity of the data distributions across clients. The smaller beta is, the more heterogeneous the distributions are.

## Starting Clients

Once the server has started and logged "FL starting," the next step, in separate terminals, is to start the three
clients. This is done by simply running (remembering to activate your environment)
```bash
python -m examples.fedrep_example.client --dataset_path /path/to/data
```
**NOTE**: The argument `dataset_path` has two functions, depending on whether the dataset exists locally or not. If
the dataset already exists at the path specified, it will be loaded from there. Otherwise, the dataset will be
automatically downloaded to the path specified and used in the run.

After the clients have been started federated learning should commence.
