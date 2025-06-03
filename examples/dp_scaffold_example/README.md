# DP-SCAFFOLD Federated Learning Example
This is an example of [Differentially Private Federated Learning on Heterogeneous Data
](https://arxiv.org/abs/2111.09278)(DP-SCAFFOLD). DP-SCAFFOLD is a differentially private adaption of SCAFFOLD - a method that corrects for client drift during the optimization procedure. In particular, DP-SCAFFOLD offers instance level privacy towards the server or a third party with access to the final model. At a given level of noise, DP-SCAFFOLD offers the same privacy guarantees as DP-FedAvg while offering better convergence. We leverage Opacus DP-SGD algorithm to impose DP guarantees and accounting is done using an instance-level privacy accountants.

In this demo, DP-SCAFFOLD is applied to an augmented version of the MNIST dataset that is non--IID. The FL server expects three clients to be spun up (i.e. it will wait until three clients report in before starting training). Each client has a modified version of the MNIST dataset. This modification essentially subsamples a certain number from the original training and validation sets of MNIST in order to synthetically induce local variations in the statistical properties of the clients training/validation data. In theory, the models should be able to perform well on their local data while learning from other clients data that has different statistical properties. The proportion of labels at each client is determined by Dirichlet distribution across the classes. The lower the beta parameter is for each class, the higher the degree of the label heterogeneity.

# Running the Example
In order to run the example, first ensure you have [installed the dependencies in your virtual environment according to the main README](/README.md#development-requirements) and it has been activated.

## Starting Server

The next step is to start the server by running
```bash
python -m examples.dp_scaffold_example.server  --config_path /path/to/config.yaml
```

## Starting Clients

Once the server has started and logged "FL starting," the next step, in separate terminals, is to start the three
clients. This is done by simply running (remembering to activate your environment)
```bash
python -m examples.dp_scaffold_example.client --dataset_path /path/to/data
```
**NOTE**: The argument `dataset_path` has two functions, depending on whether the dataset exists locally or not. If
the dataset already exists at the path specified, it will be loaded from there. Otherwise, the dataset will be
automatically downloaded to the path specified and used in the run.

After all three clients have been started, federated learning should commence.
