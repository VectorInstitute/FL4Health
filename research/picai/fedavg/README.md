# Running FedAvg Example

The following instructions outline training and validating a simple U-Net model on the Preprocessed PICAI Dataset described in the [PICAI Documentation](/research/picai/preprocessing/README.md) in a federated manner across two clients using FedAvg. The dataset is partitioned randomly in a uniform manner based on the number of clients. The provided script spins up server and clients on the same machine which is demonstrated below. See the main [PICAI Documentation](/research/picai/README.md) for instructions on creating and activating the environment required to execute the following code. The following commands can must executed from the root directory of the repository. First, spin up the server as follows:

```bash
python -m research.picai.fedavg.server --config-path path/to/config.yaml --artifact_dir path/to/artifact_dir --n_client <num_clients>
```

Then start a single or multiple clients in different sessions using the following command

```bash
python -m research.picai.fedavg.client --artifact_dir path/to/artifacts --base_dir path/to/base_dir --overviews_dir path/to/overviews_dir
```

For a complete list of arguments and their definitions, please run the scripts with the --help argument.

## Running on Vector Cluster
A slurm script has been made available to launch the experiments on the Vector Cluster. This script will automatically handle relaunching the job if it times out. The script `run_fl_single_node.slrm` first spins up a server and subsequently the clients to perform an FL experiment on the same machine. The commands below should be run from the top level directory:

```bash
sbatch research/picai/fedavg/run_fl_single_node.slrm path_to_config.yaml folder_for_server_logs/ folder_for_client_logs/ path_to_desired_venv/ <n_clients>
```
__An example__
```bash
sbatch research/picai/fedavg/run_fl_single_node.slrm research/picai/fedavg/config.yaml research/picai/fedavg/server_logs/ research/picai/fedavg/client_logs/ /h/jewtay/fl4health_env/ 2
```

__Note__: The `server_logs/` and `client_logs/` folders must already exist and this is where the python logs will be placed, capturing outputs of the training/evaluation on the servers and clients, respectively. The path `/h/jewtay/fl4health_env/` is a full path to the python venv we want to activate for the server and client python executions on each node.
