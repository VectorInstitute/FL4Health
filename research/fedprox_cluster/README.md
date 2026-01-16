# FedProx Federated Learning On The Cluster

The scripts in this folder use the client and server scripts from the [FedProx Example](/examples/fedprox_example/). For more information on that method, see the documentation in the corresponding README.md. The scripts in this folder facilitate running FL on Vector's cluster where clients have their own dedicated GPUs as well as the server. The master script is `run_fl_cluster.sh`. It is used to orchestrate the creation of the server and clients via the SLURM scripts `run_server.slrm` and `run_client.slrm`. An example of the usage is below. Note that the script needs to be run from the top level of the FL4Health repository. Moreover, a python environment with the required libraries must already exist.  See `Establishing a Python VENV on Vector's Cluster` below. The commands below should be run from the top level directory

```bash
./research/run_fl_cluster.sh server_port_number path_to_config.yaml folder_for_server_logs/ folder_for_client_logs/ path_to_desired_venv/
```
__An example__
```bash
./research/fedprox_cluster/run_fl_cluster.sh 8111 examples/fedprox_example/config.yaml research/fedprox_cluster/server_logs/ research/fedprox_cluster/client_logs/ /path/to/fl4health/.venv/
```

__Note__: The `server_logs/` and `client_logs/` folders must already exist and this is where the python logs will be placed, capturing outputs of the training/evaluation on the servers and clients, respectively. The path `/path/to/fl4health/.venv/` is a full path to the python venv we want to activate for the server and client python executions on each node (typically this will be the `.venv/` directory within your FL4Health repository).

### Establishing a Python VENV on Vector's Cluster

Navigate to the FL4Health repository directory and run

```bash
module load python/3.10.12
cd path/to/fl4health
pip install uv
uv sync --extra dev --extra test --extra codestyle
source .venv/bin/activate
```

This will create a virtual environment in `.venv/` within the repository directory.
