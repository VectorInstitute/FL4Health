# Basic Federated Learning Example with Docker
This example provides a simple implementation of federated learning on the CIFAR dataset using docker. In particular, Docker Compose is used to spin up a federated learning server with 2 clients. Each client has the same "local" dataset. I.e. they each load the complete CIFAR dataset and therefore have the same training and validation sets. The server has some custom metrics aggregation, but is otherwise a vanilla FL implementation using FedAvg as the server side optimization.
## Running Example
In order to run the demo, first ensure that Docker Desktop is running. Instruction to download Docker Desktop can be found [here](https://www.docker.com/products/docker-desktop/). Then from this directory, execute the following command:
```
docker compose up
```
This will initiate the services specified in the file `docker-compose.yml`. Namely, the fl_server and fl_client services are built and run according to the Dockerfiles in the `fl_server` and `fl_client` directories, respectively. Each of these directories also include a `requirement.txt` file separate from the `requirement.txt` in the root of the repository. These files include the python packages required to run the respective containers.

A config.yaml must be present in the root of this directory with the following arguments:
* `n_clients`: number of clients the server waits for in order to run the FL training
* `local_epochs`: number of epochs each client will train for locally
* `batch_size`: size of the batches each client will train on
* `n_server_rounds`: The number of rounds to run FL
