#!/bin/bash

###############################################
# out of date
# Usage:
#
#  ./research/flamby/fed_tcga_brca/fedavg/run_hp_sweep.sh \
#   path_to_config.yaml \
#   path_to_folder_for_artifacts/ \
#   path_to_folder_for_dataset/ \
#   path_to_desired_venv/
#
# Example:
# ./research/flamby/fed_tcga_brca/fedavg/run_hp_sweep.sh \
#   research/flamby/fed_tcga_brca/fedavg/config.yaml \
#   research/flamby/fed_tcga_brca/fedavg/ \
#   /Users/david/Desktop/FLambyDatasets/fed_tcga_brca/ \
#   /h/demerson/vector_repositories/fl4health_env/
#
# Notes:
# 1) The bash command above should be run from the top level directory of the repository.
###############################################

SERVER_CONFIG_PATH=$1
ARTIFACT_DIR=$2
DATASET_DIR=$3
VENV_PATH=$4
HYPERPARAMETER_NAME=$5

shift 5
HYPERPARAMETER_VALUES=("$@")
# FedHeartDisease LR Hyperparmeters from paper are not suitable for AdamW
# LR_VALUES=( 0.00001 0.0001 0.001 0.01 0.1 )

SERVER_PORT=8200

FOLDER="flamby_local_dp"

# Search through these hyperparameters
# HYPERPARAMETER_NAME="noise_multiplier"
# HYPERPARAMETER_VALUES=(0.001 0.01 0.1 0.5 1 10)
# HYPERPARAMETER_VALUES=(0.001 0.005 0.01 0.05 0.1 0.5)
DEFAULT_LR=0.001

# Create sweep folder
SWEEP_DIRECTORY="${ARTIFACT_DIR}hp_sweep_results"
echo "Creating sweep folder at ${SWEEP_DIRECTORY}"
mkdir ${SWEEP_DIRECTORY}

if [ "$HYPERPARAMETER_NAME" = "noise_multiplier" ]; then
  ALIAS="noise"
else 
  ALIAS=$HYPERPARAMETER_NAME
fi

for HYPERPARAMETER_VALUE in "${HYPERPARAMETER_VALUES[@]}";
do
  EXPERIMENT_NAME="${ALIAS}_${HYPERPARAMETER_VALUE}"
  echo "Beginning Experiment ${EXPERIMENT_NAME}"
  EXPERIMENT_DIRECTORY="${SWEEP_DIRECTORY}/${EXPERIMENT_NAME}/"
  echo "Creating experiment folder ${EXPERIMENT_DIRECTORY}"
  mkdir "${EXPERIMENT_DIRECTORY}"
  SERVER_ADDRESS="0.0.0.0:${SERVER_PORT}"
  echo "Server Address: ${SERVER_ADDRESS}"
  SBATCH_COMMAND="research/${FOLDER}/fed_tcga_brca/run_fold_experiment.slrm \
    ${SERVER_CONFIG_PATH} \
    ${EXPERIMENT_DIRECTORY} \
    ${DATASET_DIR} \
    ${VENV_PATH} \
    ${DEFAULT_LR} \
    ${SERVER_ADDRESS} \
    ${HYPERPARAMETER_NAME} \
    ${HYPERPARAMETER_VALUE}"

  echo "Running sbatch command ${SBATCH_COMMAND}"
  sbatch ${SBATCH_COMMAND}
  ((SERVER_PORT=SERVER_PORT+1))
done

echo Experiments Launched
