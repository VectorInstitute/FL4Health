# Federated Learning with nnUNet

## Usage
To train nnunet models using federated learning first set up a config yaml file that has the following keys. Note that either local_epochs or local_steps can be used but not both.

```yaml
n_clients: 1
nnunet_config: 2d
nnunet_plans: /path/to/nnunet/plans/file.json
fold: 0 # Which fold of the data to use for validation
n_server_rounds: 1 # number of server rounds
local_epochs: 1 # number of epochs per server round
server_address: '0.0.0.0:8080' # Default is server is the same machine
starting_checkpoint: /home/shawn/Code/nnunet_storage/nnUNet_results/Dataset012_PICAI-debug/nnUNetTrainer_1epoch__nnUNetPlans__2d/fold_0/checkpoint_best.pth # This is currently required due to a 'bug' in flwr. I have raised an issue: https://github.com/adap/flower/issues/3770
```

After creating a config file start a server using the following command. Ensure your virtual environment has been properly set up using poetry and that you have included the 'picai' group in ```poetry install```

```bash
python -m research.picai.fl_nnunet.start_server --config-path path/to/config.yaml
```

Then start a single or multiple clients in different sessions using the following command

```bash
python -m research.picai.fl_nnunet.start_client --dataset-id 012
```

The federated training will commmence once n_clients have been instantiated.

## Running on Vector Cluster
A slurm script has been made available to launch the experiments on the Vector Cluster. This script will automatically handle relaunching the job if it times out. The script `run_fl_single_node.slrm` first spins up a server and subsequently the clients to perform an FL experiment on the same machine. The commands below should be run from the top level directory:

```bash
sbatch research/picai/fl_nnunet/run_fl_single_node.slrm path_to_config.yaml path_to_desired_venv/ <n_clients> <fold> <dataset-id>
```
__An example__
```bash
sbatch research/picai/fl_nnunet/run_fl_single_node.slrm research/picai/fl_nnunet/config.yaml /h/jewtay/fl4health_env/ 2 0 005
```

__Note__: The path `/h/jewtay/fl4health_env/` is a full path to the python venv we want to activate for the server and client python executions on each node. Artifacts from the experiment will be stored at /experiment/${USER}/$SLURM_JOB_ID.
