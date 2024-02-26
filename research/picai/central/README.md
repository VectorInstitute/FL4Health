# Running Centralized Example

The following instructions outline training and validating a simple U-Net model on the Preprocessed Dataset described in the [PICAI Documentation](/research/picai/README.md) using a centralized setup. An example of the usage is below. Note that the script needs to be run from the top level of the FL4Health repository. Moreover, a python environment with the required libraries must already exist.  See the main PICAI documentation Cluster [PICAI Documentation](/research/picai/README.md) for instructions on creating and activating environment required to exectute the following code. The commands below should be run from the top level directory:

```bash
./research/picai/central/launch.slrm folder_for_artifacts/ path_to_desired_venv/ fold_id
```
__An example__
```bash
./research/picai/central/launch.slrm research/picai/central/artifacts/ /h/jewtay/fl4health_env/ 0
```
where fold_id is an integer from 0 - 4.

__Note__: The `artifacts/` folder must already exist and this is where the python logs will be placed, capturing outputs of the training/evaluation on the servers and clients, respectively. The path `/h/jewtay/fl4health_env/` is a full path to the python venv we want to activate for the server and client python executions on each node.
