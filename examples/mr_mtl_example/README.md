# MR-MTL Federated Learning Example

In this example, we implement the MR-MTL algorithm from the paper [MR-MTL: On Privacy and Personalization in Cross-Silo Federated Learning](https://arxiv.org/pdf/2206.07902.pdf). The method is somewhat related to FedProx and Ditto. Essentially, at the start of each client training round we do not update local model weights directly with the global model weights, but instead we constrain the model weights to be close to the initial global model weights computed by averaging the model weights at the end of the previous round. Such mean-regularized training is done by adding a penalty term to the loss function that constrains the local model weights to be close to the initial global model weights. The strength of the penalty term is controlled by a hyperparameter $\lambda > 0$. To be more specific, the local model is trained using the same loss function as in FedAvg, but with the penalty term added. The penalty term is added to the loss function as follows:
$$
l_{w} + \lambda \Vert w - w_g \Vert_2^2,
$$
where $w_g$ is the set of weights for the global model at the start of the current client training round.

In this demo, MR-MTL is applied to a modified version of the MNIST dataset that is non--IID. The FL server expects three clients to be spun up (i.e. it will wait until three clients report in before starting training). Each client has it's own modified version of the MNIST dataset. This modification essentially subsamples a certain number from the original training and validation sets of MNIST in order to synthetically induce local variations in the statistical properties of the clients training/validation data. In theory, the models should be able to perform well on their local data while learning from other clients data that has different statistical properties. The proportion of labels at each client is determined by Dirichlet distribution across the classes. The lower the beta parameter is for each class, the higher the degree of the label heterogeneity.

The server has some custom metrics aggregation and but other than that uses FedAvg as its server-side optimization.

### Running the Example

In order to run the example, first ensure you have [installed the dependencies in your virtual environment according to the main README](/README.md#development-requirements) and it has been activated.

## Starting Server

The next step is to start the server by running
```bash
python -m examples.mr_mtl_example.server --config_path /path/to/config.yaml
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
python -m examples.mr_mtl_example.client --dataset_path /path/to/data

# alternatively, with subclass of FlexibleClient
python -m examples.mr_mtl_example.client_flexible --dataset_path /path/to/data
```
**NOTE**: The argument `dataset_path` has two functions, depending on whether the dataset exists locally or not. If
the dataset already exists at the path specified, it will be loaded from there. Otherwise, the dataset will be
automatically downloaded to the path specified and used in the run.

After all three clients have been started, federated learning should commence.
