# Ditto Federated Learning Example

In this example, we implement the Ditto algorithm from the paper [Ditto: Fair and Robust Federated Learning Through Personalization
](https://arxiv.org/abs/2012.04221). This method has been shown to perform quite well on certain benchmarks (though not as well as others). See for example [Benchmark for Personalized Federated Learning](https://www.computer.org/csdl/journal/oj/2024/01/10316561/1S2UbvQk5Tq) and [pFL-Bench: A Comprehensive Benchmark for Personalized Federated Learning](https://arxiv.org/abs/2206.03655). The method is somewhat related to FedProx. Essentially, the model trains a global model through FedAvg and uses that global model, at each round to also train a personalized model. That model is trained using the same loss function, but with the initial global model (at the start of each round) as a penalty. That is, the local model weights are constrained using $\lambda > 0$ such that
$$
l_{w} + \lambda \Vert w - w_g \Vert_2^2,
$$
where $w_g$ is the set of weights for the global model at the start of the current client training round.

In this demo, Ditto is applied to a modified version of the MNIST dataset that is non--IID. The FL server expects three clients to be spun up (i.e. it will wait until three clients report in before starting training). Each client has it's own modified version of the MNIST dataset. This modification essentially subsamples a certain number from the original training and validation sets of MNIST in order to synthetically induce local variations in the statistical properties of the clients training/validation data. In theory, the models should be able to perform well on their local data while learning from other clients data that has different statistical properties. The proportion of labels at each client is determined by Dirichlet distribution across the classes. The lower the beta parameter is for each class, the higher the degree of the label heterogeneity.

The server has some custom metrics aggregation and but other than that uses FedAvg as its server-side optimization.

### Running the Example

In order to run the example, first ensure you have [installed the dependencies in your virtual environment according to the main README](/README.md#development-requirements) and it has been activated.

## Starting Server

The next step is to start the server by running
```bash
python -m examples.ditto_example.server --config_path /path/to/config.yaml
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
# run a ditto client that's built with `DittoClient`
python -m examples.ditto_example.client --dataset_path /path/to/data

# Or, run a ditto client that's built via an application of `make_it_personal` on a `BasicClient`.
python -m examples.ditto_example.client_dynamic --dataset_path /path/to/data
```
**NOTE**: The argument `dataset_path` has two functions, depending on whether the dataset exists locally or not. If
the dataset already exists at the path specified, it will be loaded from there. Otherwise, the dataset will be
automatically downloaded to the path specified and used in the run.

After all three clients have been started, federated learning should commence.
