# Running FedAvg Example

The following instructions outline training and validating a simple U-Net model on the Preprocessed PICAI Dataset described in the [PICAI Documentation](/research/picai/README.md) in a federated manner across two clients using FedAvg. Two scripts are provided, one in which the server and clients are spun up on the same machine and one in which the server and each serperate client is spun up on a different machine along with example invocations below. A python environment with the required libraries must already exist.  See the main PICAI documentation Cluster [PICAI Documentation](/research/picai/README.md) for instructions on creating and activating environment required to exectute the following code.

## Server and Client on Seperate Machines
The script `run_fl_cluster.sh` is used to orchestrate the submission of the server and client jobs to the cluster via the SLURM scripts `run_server.slrm` and `run_client.slrm`. An example of the usage is below. The commands below should be run from the top level directory:

```bash
./research/picai/fedavg/run_fl_cluster.sh server_port_number path_to_config.yaml folder_for_server_logs/ folder_for_client_logs/ path_to_desired_venv/
```
__An example__
```bash
./research/picai/fedavg/run_fl_cluster.sh 8111 research/picai/fedavg/config.yaml research/picai/fedavg/server_logs/ research/picai/fedavg/client_logs/ /h/jewtay/fl4health_env/
```

__Note__: The `server_logs/` and `client_logs/` folders must already exist and this is where the python logs will be placed, capturing outputs of the training/evaluation on the servers and clients, respectively. The path `/h/jewtay/fl4health_env/` is a full path to the python venv we want to activate for the server and client python executions on each node.

## Server and Client on Same Machine
The script `run_fl_single_node.slrm` first spins up a server and subsequently the clients to perform an FL experiment on the same machine. The commands below should be run from the top level directory:

```bash
./research/picai/fedavg/run_fl_single_node.slrm path_to_config.yaml folder_for_server_logs/ folder_for_client_logs/ path_to_desired_venv/
```
__An example__
```bash
./research/picai/fedavg/run_fl_single_node.slrm research/picai/fedavg/config.yaml research/picai/fedavg/server_logs/ research/picai/fedavg/client_logs/ /h/jewtay/fl4health_env/
```

__Note__: The `server_logs/` and `client_logs/` folders must already exist and this is where the python logs will be placed, capturing outputs of the training/evaluation on the servers and clients, respectively. The path `/h/jewtay/fl4health_env/` is a full path to the python venv we want to activate for the server and client python executions on each node.
