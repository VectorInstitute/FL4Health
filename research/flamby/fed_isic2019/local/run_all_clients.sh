#!/bin/bash

###############################################
# Usage:
#
#  ./research/flamby/fed_isic2019/local/run_all_clients.sh \
#   path_to_folder_for_artifacts/ \
#   path_to_folder_for_dataset/ \
#   path_to_desired_venv/
#
# Example:
# ./research/flamby/fed_isic2019/local/run_all_clients.sh \
#   research/flamby/fed_isic2019/local/ \
#   /Users/xxx/Desktop/FLambyDatasets/fedisic2019/ \
#   /h/xxx/vector_repositories/fl4health_env/
#
# Notes:
# 1) The bash command above should be run from the top level directory of the repository.
###############################################

ARTIFACT_DIR=$1
DATASET_DIR=$2
VENV_PATH=$3

# FedIsic has a total of 6 clients
CLIENT_NUMBERS=( 0 1 2 3 4 5 )

# Create sweep folder
SWEEP_DIRECTORY="${ARTIFACT_DIR}client_sweep_results"
echo "Creating sweep folder at ${SWEEP_DIRECTORY}"
mkdir ${SWEEP_DIRECTORY}

for CLIENT_NUMBER in "${CLIENT_NUMBERS[@]}";
do
  EXPERIMENT_NAME="client_${CLIENT_NUMBER}"
  echo "Beginning Experiment ${EXPERIMENT_NAME}"
  EXPERIMENT_DIRECTORY="${SWEEP_DIRECTORY}/${EXPERIMENT_NAME}/"
  echo "Creating experiment folder ${EXPERIMENT_DIRECTORY}"
  mkdir "${EXPERIMENT_DIRECTORY}"
  SBATCH_COMMAND="research/flamby/fed_isic2019/local/run_fold_experiment.slrm \
    ${EXPERIMENT_DIRECTORY} \
    ${DATASET_DIR} \
    ${VENV_PATH} \
    ${CLIENT_NUMBER}"
  echo "Running sbatch command ${SBATCH_COMMAND}"
  sbatch ${SBATCH_COMMAND}
done

echo Experiments Launched
