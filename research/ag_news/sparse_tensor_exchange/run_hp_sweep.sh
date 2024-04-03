#!/bin/bash

###############################################
# Usage:
#
#  ./research/ag_news/sparse_tensor_exchange/run_hp_sweep.sh \
#   path_to_config.yaml \
#   path_to_folder_for_artifacts/ \
#   path_to_folder_for_dataset/ \
#   path_to_desired_venv/
#
# Example:
# ./research/ag_news/sparse_tensor_exchange/run_hp_sweep.sh \
#   research/ag_news/sparse_tensor_exchange/config.yaml \
#   research/ag_news/sparse_tensor_exchange/ \
#   /Users/david/Desktop/AG_NEWS/ \
#   /h/demerson/vector_repositories/fl4health_env/
#
# Notes:
# 1) The bash command above should be run from the top level directory of the repository.
###############################################

SERVER_CONFIG_PATH=$1
ARTIFACT_DIR=$2
DATASET_DIR=$3
VENV_PATH=$4

# LR_VALUES=( 0.00001 0.0001 0.001 0.01 0.1 )
LR_VALUES=( 0.01 )

SPARSITY_LEVELS=( 0.4 )

SERVER_PORT=8100

# Create sweep folder
SWEEP_DIRECTORY="${ARTIFACT_DIR}hp_sweep_results"
echo "Creating sweep folder at ${SWEEP_DIRECTORY}"
mkdir ${SWEEP_DIRECTORY}

for SPARSITY_LEVEL in "${SPARSITY_LEVELS[@]}";
do
  for LR_VALUE in "${LR_VALUES[@]}";
  do
    EXPERIMENT_NAME="sparsity_${SPARSITY_LEVEL}_lr_${LR_VALUE}"
    echo "Beginning Experiment ${EXPERIMENT_NAME}"
    EXPERIMENT_DIRECTORY="${SWEEP_DIRECTORY}/${EXPERIMENT_NAME}/"
    echo "Creating experiment folder ${EXPERIMENT_DIRECTORY}"
    mkdir "${EXPERIMENT_DIRECTORY}"
    SERVER_ADDRESS="0.0.0.0:${SERVER_PORT}"
    echo "Server Address: ${SERVER_ADDRESS}"
    SBATCH_COMMAND="research/ag_news/sparse_tensor_exchange/run_fold_experiment.slrm \
      ${SERVER_CONFIG_PATH} \
      ${EXPERIMENT_DIRECTORY} \
      ${DATASET_DIR} \
      ${VENV_PATH} \
      ${LR_VALUE} \
      ${SPARSITY_LEVEL} \
      ${SERVER_ADDRESS}"
    echo "Running sbatch command ${SBATCH_COMMAND}"
    sbatch ${SBATCH_COMMAND}
    ((SERVER_PORT=SERVER_PORT+1))
  done
done

echo Experiments Launched
