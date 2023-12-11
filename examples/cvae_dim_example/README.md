# Federated Dimensionality reduction using CVAEs Example
In this example, clients first reduce the dimensionality of the data using an already federally trained conditional variational auto-encoder (please refer to the CVAE example). Then, the data in the reduced dimension is used to train a small neural network model (`mnist_model.py`). Clients in this example inherit from a `CVAETrainer` class to use the CVAE processing functionalities i.e. set the client-specific condition and reduce the dimensionality of data via CVAE data pre-processor. Federated training for the main task of MNIST classification can be done with any of the FL approaches by specifying the parent client class. The default parent client class in this example is the `BasicClient` and the server uses Federated Averaging.

## Running the Example
In order to run the example, first ensure you have the virtual env of your choice activated and run
```
pip install --upgrade pip
pip install -r requirements.txt
```
to install all of the dependencies for this project.

## Starting Server

The next step is to start the server by running
```
python -m examples.cvae_dim_example.server  --config_path /path/to/config.yaml
```
from the FL4Health directory. The following arguments must be present in the specified config file:
* `n_clients`: number of clients the server waits for in order to run the FL training
* `local_epochs`: number of epochs each client will train for locally
* `batch_size`: size of the batches each client will train on
* `n_server_rounds`: The number of rounds to run FL
* `checkpoint_path`: path to save the best server model
* `latent_dim`: size of the latent vector in the CVAE or VAE model
* `CVAE_model_path`: path to the saved CVAE model for dimesionality reduction

## Starting Clients

Once the server has started and logged "FL starting," the next step, in separate terminals, is to start the two
clients. This is done by simply running (remembering to activate your environment)
```
python -m examples.CVAE_example.client --dataset_path /path/to/data --condition "client's ID number"
```
**NOTE**: The argument `dataset_path` has two functions, depending on whether the dataset exists locally or not. If
the dataset already exists at the path specified, it will be loaded from there. Otherwise, the dataset will be
automatically downloaded to the path specified and used in the run.

After both clients have been started federated learning should commence.

**NOTE**: In this example, the argument `condition ` is used to set a client-specific condition on the CVAE model. Here, client IDs are used as the condition on their data. Another option would be to use the  "label" as a condition, which prompts the clients to condition their data based on the target of each sample. In this example, we are just using the CVAE to encode the data, therefore you need to make sure to use the same condition as the one used during the training.

If you choose to set `--condition 'label'`, don't forget to also adjust the `num_conditions` variable in the config file to correspond to the number of the classes in the data.

### In this example
You can use the following commands to run the clients. 
Client 0:  
```
python -m examples.cvae_dim_example.client --dataset_path examples/datasets/MNIST --condition "0"
```
Client 1:
```
python -m examples.cvae_dim_example.client --dataset_path examples/datasets/MNIST --condition "1"
```