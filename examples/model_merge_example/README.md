# Model Merge Example
This example provides an illustration of a model merging setup on the MNIST dataset in which clients each have
a copy of the same architecture with different weights initialized via local pre-training. The goal is to
average these weights and perform evaluation on the client side and the server side with the provided
evaluation function. The server expects two clients to be spun up (i.e. it will wait until two clients
report in before starting model merging and evaluation). For convenience, pre-trained models on the MNIST
train set have been provided for each of the clients in `/examples/assets/model_merge_example/`
under `0.pt` and `1.pt`. The model merging and subsequent evaluation can be performed with these weights
out-of-the-box.

## Running the Example
In order to run the example, first ensure you have [installed the dependencies in your virtual environment according to the main README](/README.md#development-requirements) and it has been activated.

## Starting Server

The next step is to start the server by running:
```bash
python -m examples.model_merge_example.server --config_path /path/to/config
```
Optionally, you can provide a path to an evaluation dataset (`--data_path`) to evaluate the merged models on the
server side.

For a full list of arguments and their definitions: `python -m examples.model_merge_example.server --help`

## Starting Clients

Once the server has started and logged "FL starting," the next step, in separate terminals, is to start the two
clients. This is done by simply running (remembering to activate your environment)
```bash
python -m examples.basic_example.client --dataset_path "examples/datasets/mnist_data/" --model_path "/path/to/checkpoint.pt"
```
For a full list of arguments and their definitions: `python -m examples.model_merge_example.client --help`
