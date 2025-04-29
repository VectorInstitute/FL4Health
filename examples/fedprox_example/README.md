# FedProx Federated Learning Example

In this example, we implement the FedProx algorithm from the paper [Federated Optimization in Heterogeneous Networks](https://arxiv.org/pdf/1812.06127.pdf). This method is one of the standard baseline "personalized" extensions of vanilla federated learning (FL) algorithm to help deal with heterogeneity in local training datasets of each client in the FL learning system. The basic idea of the algorithm is quite straightforward. The server side aggregation/optimization step remains the same. That is, standard FedAvg or any other server-side method is used without modification. On the client side, the local objective function is modified to include a "proximal" loss term, which amounts to a penalty for large local weight deviations from the original global weights at each server-side iteration. Say that $t$ represents the current server round and $w_t$ are the aggregated weights received by each client to being local training with. For loss function $l_{w}$, for local weights $w$ and positive scalar $\mu \geq 0$, the proximal loss is
$$
l_{w} + \frac{\mu}{2} \Vert w - w_t \Vert_2^2,
$$
where $\Vert \cdot \Vert_2$ is the $l_2$ norm. This term essentially restricts the update magnitude proposed by any one client participating in a round of training.

In this demo, FedProx is applied to a modified version of the MNIST dataset that is non--IID. The FL server expects three clients to be spun up (i.e. it will wait until three clients report in before starting training). Each client has it's own modified version of the MNIST dataset. This modification essentially subsamples a certain number from the original training and validation sets of MNIST in order to synthetically induce local variations in the statistical properties of the clients training/validation data. In theory, the models should be able to perform well on their local data while learning from other clients data that has different statistical properties. The proportion of labels at each client is determined by Dirichlet distribution across the classes. The lower the beta parameter is for each class, the higher the degree of the label heterogeneity.

The server has some custom metrics aggregation and uses FedProx as its server-side optimization.

### Weights and Biases Reporting

This example is also capable of logging results to your Weights and Biases account by including the correct components under the `reporting_config` section in the `config.yaml`. You'll also need to set the `entity` value to your Weights and Biases entity. Once those two things are set, you should be able to run the example and log the results to W and B directly.

### Running the Example

In order to run the example, first ensure you have [installed the dependencies in your virtual environment according to the main README](/README.md#development-requirements) and it has been activated.

## Starting Server

The next step is to start the server by running
```bash
python -m examples.fedprox_example.server --config_path /path/to/config.yaml
```
from the FL4Health directory. The following arguments must be present in the specified config file:
* `n_clients`: number of clients the server waits for in order to run the FL training
* `local_epochs`: number of epochs each client will train for locally
* `batch_size`: size of the batches each client will train on
* `n_server_rounds`: The number of rounds to run FL

## Starting Clients

Once the server has started and logged "FL starting," the next step, in separate terminals, is to start the three
clients. This is done by simply running (remembering to activate your environment)
```bash
python -m examples.fedprox_example.client --dataset_path /path/to/data
```
**NOTE**: The argument `dataset_path` has two functions, depending on whether the dataset exists locally or not. If
the dataset already exists at the path specified, it will be loaded from there. Otherwise, the dataset will be
automatically downloaded to the path specified and used in the run.

After all three clients have been started, federated learning should commence.
