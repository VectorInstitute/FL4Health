# Running Centralized Example

The following instructions outline training and validating a simple U-Net model on the Preprocessed Dataset described in the [PICAI Documentation](/research/picai/README.md) using a centralized setup. Running the centralized example out of the box is as simple as executing the command below.

An example of the usage is below. Note that the script needs to be run from the top level of the FL4Health repository. Moreover, a python environment with the required libraries must already exist.  See the main PICAI documentation Cluster [PICAI Documentation](/research/picai/README.md) for instructions on creating and activating the environment required to execute the following code. The commands below should be run from the top level directory:

```bash
python research/picai/central/train.py --overviews_dir /path/to/overviews_dir --fold <fold_num> --run_name <run_name>
```

For a full list of arguments and their definitions, run `python research/picai/central/train.py --help`

## Running on Vector Cluster
A slurm script has been made available to launch the experiments on the Vector Cluster. This script will automatically handle relaunching the job if it times out. The slurm script can be used as follows:

```bash
sbatch research/picai/central/launch.slrm folder_for_artifacts/ path_to_desired_venv/ fold_id run_name
```
where fold_id is an integer from 0 - 4.
__An example__
```bash
sbatch research/picai/central/launch.slrm research/picai/central/artifacts/ /h/jewtay/fl4health_env/ 0 test_run
```

__Note__: The `artifacts/` folder must already exist and this is where the python logs will be placed, capturing outputs of the training/evaluation on the servers and clients, respectively. The path `/h/jewtay/fl4health_env/` is a full path to the python venv we want to activate for the server and client python executions on each node.
