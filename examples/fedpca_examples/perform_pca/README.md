# Federated Principal Component Analysis Example
This example performs federated principal component analysis. The goal is to compute the principal components of a subset of the MNIST dataset, under the condition that the data is distributed across four distinct clients.

This is achieved by each client performing PCA locally at first, then the principal components are sent to a central server to be merged.

Each client performs Dirichlet subsampling on the whole dataset to produce heterogeneous local datasets. This is done to ensure that local principal components are distinct across different clients.

## Running the Example
In order to run the example, first ensure you have [installed the dependencies in your virtual environment according to the main README](/README.md#development-requirements) and it has been activated.

## Starting Server

The next step is to start the server by running
```bash
python -m examples.fedpca_examples.perform_pca.server  --config_path /path/to/config.yaml
```
from the FL4Health directory. The following arguments must be present in the specified config file:
* `n_clients`: number of clients the server waits for in order to run the FL training
* `batch_size`: size of the batches each client will train on
* `low_rank`: whether the data matrix is low rank or not. If the user has prior knowledge that the data matrix is low rank, then this parameter can be set to True so that each client can leverage this knowledge to perform local PCA more efficiently.
* `full_svd`: determines whether full SVD or reduced SVD is performed by each client.
* `rank_estimation`: an estimation of the rank of the data matrix. This is only used if `low_rank` is set to True.
* `center_data`: if set to True, the mean of data will be subtracted from all data points before local PCA is performed.
* `num_components`: by default, after merging is completed, each client will use the merged principal components to compute the reconstruction error on its validation set. This parameter specifies the number of principal components that are used in this evaluation process.

## Starting Clients

Once the server has started and logged "FL starting," the next step, in separate terminals, is to start the four
clients. This is done by simply running (remembering to activate your environment)
```bash
python -m examples.fedpca_examples.perform_pca.client --dataset_path /path/to/data --components_save_dir /dir/to/save/components
```
**NOTE**:

* The argument `dataset_path` has two functions, depending on whether the dataset exists locally or not. If
the dataset already exists at the path specified, it will be loaded from there. Otherwise, the dataset will be
automatically downloaded to the path specified and used in the run.

* The argument `components_save_dir` specifies the directory in which the merged principal components will be saved, so they can be leveraged for other downstream tasks. An example of dimensionality reduction can be found at `examples/fedpca_examples/dim_reduction`.

After the clients have been started federated pca should commence.
