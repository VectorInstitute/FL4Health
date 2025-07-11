# FedBN Federated Learning Example

In this example, we implement the FedBN algorithm from the paper [FedBN: Federated Learning on Non-IID Features via Local Batch Normalization](https://arxiv.org/abs/2102.07623). This method is fairly straightforward and represents only a slight modification of the standard FedAvg approach for model networks that incorporate batch normalization layers. In this method, the server and clients exchange all parameters except for those associated with BatchNormalization. This implies that all model parameters are globally aggregated __except__ those underlying each clients BatchNormalization layers. The authors posit that this allows each client's model to adapt to local conditions while still learning from global datasets via aggregation of all other layers.

__NOTE__: This method doesn't do anything for models that do not significantly incorporate BN layers. Moreover, the authors show that this method performs better than FedAvg. However, it is unclear if the authors accumulate state in these layers which are aggregated by FedAvg. This could also be a source of FedAvg's under-performance, as weighted averaging of variance estimates isn't quite the right thing to do.

In any case, the method is very simple to implement in our framework. One simply needs to use a parameter exchanger with layer exclusions and specify that you'd like to exclude exchange of BatchNormalization layers. In this demo, FedBN is applied to a modified version of the MNIST dataset that is non--IID. The FL server expects three clients to be spun up (i.e. it will wait until three clients report in before starting training). Each client has it's own modified version of the MNIST dataset. This modification essentially subsamples a certain number from the original training and validation sets of MNIST in order to synthetically induce local variations in the statistical properties of the clients training/validation data. In theory, the models should be able to perform well on their local data while learning from other clients data that has different statistical properties. The proportion of labels at each client is determined by Dirichlet distribution across the classes. The lower the beta parameter is for each class, the higher the degree of the label heterogeneity.

The server has some custom metrics aggregation and uses Federated Averaging as its server-side optimization.

### Running the Example

In order to run the example, first ensure you have [installed the dependencies in your virtual environment according to the main README](/README.md#development-requirements) and it has been activated.

## Starting Server

The next step is to start the server by running
```bash
python -m examples.fedbn_example.server --config_path /path/to/config.yaml
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
python -m examples.fedbn_example.client --dataset_path /path/to/data
```
**NOTE**: The argument `dataset_path` has two functions, depending on whether the dataset exists locally or not. If
the dataset already exists at the path specified, it will be loaded from there. Otherwise, the dataset will be
automatically downloaded to the path specified and used in the run.

The argument `dataset_name` has the following specifications:
   - **Purpose**: Specifies the name of the dataset to be used.
   - **Functionality**: Determines the type of client to be initialized based on the dataset name. For instance, `SkinCancerFedBNClient` is used for skin cancer datasets, while `MnistFedBNClient` is used for the MNIST dataset.
   - **Supported Values**:
     - Skin Cancer Datasets: `"Barcelona"`, `"Rosendahl"`, `"Vienna"`, `"UFES"`, `"Canada"`
     - MNIST Dataset: `"mnist"`
   - **Default Value**: `"mnist"`
   - **Usage**: `--dataset_name <dataset_name>`

For more details on the skin cancer datasets, please refer to the README file located at [Skin Cancer Dataset README](../../datasets/skin_cancer/README.md).

After all three clients have been started, federated learning should commence.
