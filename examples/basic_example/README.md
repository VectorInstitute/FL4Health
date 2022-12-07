# Basic Federated Learning Example
This example provides an very simple implementation of a federated learning training setup on the CIFAR dataset. The
FL server expects two clients to be spun up (i.e. it will wait until two clients report in before starting training).
Each client has the same "local" dataset. I.e. they each load the complete CIFAR dataset and therefore have the same
training and validation sets. The server has some custom metrics aggregation, but is otherwise a vanilla FL
implementation using FedAvg as the server side optimization.

# Running the Example
In order to run the example, first ensure you have the virtual env of your choice activated and run
```
pip install --upgrade pip
pip install -r requirements.txt
```
to install all of the dependencies for this project.

## Starting Server

The next step is to start the server by running
```
python -m examples.basic_example.server
```
from the FL4Health directory.

## Starting Clients

Once the server has started and logged "FL starting," the next step, in separate terminals, is to start the two
clients. This is done by simply running (remembering to activate your environment)
```
python -m examples.basic_example.client
```
After both clients have been started federated learning should commence.
