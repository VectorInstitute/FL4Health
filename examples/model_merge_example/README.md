# Model Merge Example
This example provides an example of a model merging setup on the MNIST dataset in which clients each have
a copy of the same architecture with different weights initialized via local training. The goal is to
average these weights and perform evaluation on the client side and the server side with the provided
evaluation function. The server expects two clients to be spun up (i.e. it will wait until two clients
report in before starting training). In order to perform model merging, the provided script to first train
the models for each client must be run. The models are trained on the train set and evaluated on the test
set during model merging.

## Running the Example
In order to run the example, first ensure you have [installed the dependencies in your virtual environment according to the main README](/README.md#development-requirements) and it has been activated.

## Train Client Models
The following script will train a model for each client and save a corresponding checkpoint.
```
python -m examples.model_merge_example.train_silo
```
For a full list of arguments and their definitions: `python -m examples.model_merge_example.train_silo --help`
After this finishes, the model merging example can be run.

## Starting Server

The next step is to start the server by running:
```
python -m examples.model_merge_example.server
```
For a full list of arguments and their definitions: `python -m examples.model_merge_example.server --help`

## Starting Clients

Once the server has started and logged "FL starting," the next step, in separate terminals, is to start the two
clients. This is done by simply running (remembering to activate your environment)
```
python -m examples.basic_example.client --model_path /path/to/checkpoint.pt
```
For a full list of arguments and their definitions: `python -m examples.model_merge_example.client --help`
