# PCA Dimensionality Reduction Example
This example leverages federally computed principal components of the MNIST dataset to perform dimensionality reduction on the images, before proceeding with normal training.

This example assumes that the principal components of MNIST have already been computed and saved (run the example in `examples/fedpca_examples/perform_pca` to do this), and the user supplies a path to the saved principal components to perform dimensionality reduction.

Each client performs Dirichlet subsampling on the whole dataset to produce heterogeneous local datasets.

## Running the Example
In order to run the example, first ensure you have [installed the dependencies in your virtual environment according to the main README](/README.md#development-requirements) and it has been activated.

## Starting Server

The next step is to start the server by running
```bash
python -m examples.fedpca_examples.dim_reduction.server  --config_path /path/to/config.yaml
```
from the FL4Health directory. The following arguments must be present in the specified config file:
* `n_clients`: number of clients the server waits for in order to run the FL training.
* `local_epochs`: number of local epochs each client will train.
* `batch_size`: size of the batches each client will train on.
* `new_dimension`: new dimension after reduction.
* `pca_path`: path to pre-computed principal components.
* `checkpoint_path`: path where the model weights are saved.
## Starting Clients

Once the server has started and logged "FL starting," the next step is to start three
clients in separate terminals. This is done by simply running (remembering to activate your environment)
```bash
python -m examples.fedpca_examples.dim_reduction.client --dataset_path /path/to/data
```
**NOTE**:

* The argument `dataset_path` has two functions, depending on whether the dataset exists locally or not. If
the dataset already exists at the path specified, it will be loaded from there. Otherwise, the dataset will be
automatically downloaded to the path specified and used in the run.

After the clients have been started federated learning should commence.
