# Running FedAvg Example

The following instructions outline training and validating a simple U-Net model on the Preprocessed PICAI Dataset described in the [PICAI Documentation](/research/picai/README.md) in a federated manner across two clients using FedAvg. The dataset is partitioned randomly in a uniform manner based on the number of clients. The provided script spins up server and clients on the same machine which is demonstrated below. The script will automatically handle relaunching jobs that timeout. A python environment with the required libraries must already exist and a path to a configuration file must be specified.  See the main [PICAI Documentation](/research/picai/README.md) for instructions on creating and activating environment required to exectute the following code.

## Server and Client on Same Machine
The script `run_fl_single_node.slrm` first spins up a server and subsequently the clients to perform an FL experiment on the same machine. The commands below should be run from the top level directory:

```bash
sbatch research/picai/fedavg/run_fl_single_node.slrm path_to_config.yaml folder_for_server_logs/ folder_for_client_logs/ path_to_desired_venv/ n_clients
```
__An example__
```bash
sbatch research/picai/fedavg/run_fl_single_node.slrm research/picai/fedavg/config.yaml research/picai/fedavg/server_logs/ research/picai/fedavg/client_logs/ /h/jewtay/fl4health_env/ 2
```

__Note__: The `server_logs/` and `client_logs/` folders must already exist and this is where the python logs will be placed, capturing outputs of the training/evaluation on the servers and clients, respectively. The path `/h/jewtay/fl4health_env/` is a full path to the python venv we want to activate for the server and client python executions on each node.
