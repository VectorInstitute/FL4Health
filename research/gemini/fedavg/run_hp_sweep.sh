#!/bin/bash

###############################################
# Usage:
#
#  ./gemini_fl/FedAvg/run_hp_sweep.sh \
#   path_to_config.yaml \
#   path_to_folder_for_artifacts/ \
#   task ("mortality")/ \
#   number_clients/ \
#
# Example:
# ./gemini_fl/FedAvg/run_hp_sweep.sh \
#   FedAvg/config.yaml \
#   FedAvg/2_client_results/ \
#   "mortality" \
#   2 \
#
# Notes:
# 1) The bash command above should be run from the top level directory of the repository.
###############################################

SERVER_CONFIG_PATH=$1
ARTIFACT_DIR=$2
TASK_TYPE=$3
N_CLIENTS=$4

LR_VALUES=( 0.000001 0.00001 0.0001 0.001 0.01)

SERVER_PORT=8080

# Create sweep folder
SWEEP_DIRECTORY=""${ARTIFACT_DIR}hp_sweep_results""
echo "Creating sweep folder at ${SWEEP_DIRECTORY}"
mkdir ${SWEEP_DIRECTORY}
echo "Task: ${TASK_TYPE}"


for LR_VALUE in "${LR_VALUES[@]}";
do
EXPERIMENT_NAME="lr_${LR_VALUE}"
echo "Beginning Experiment ${EXPERIMENT_NAME}"
EXPERIMENT_DIRECTORY="${ARTIFACT_DIR}hp_sweep_results/${EXPERIMENT_NAME}/"
echo "Creating experiment folder ${EXPERIMENT_DIRECTORY}"
mkdir "${EXPERIMENT_DIRECTORY}"
SERVER_ADDRESS="0.0.0.0:${SERVER_PORT}"

echo "Server Address: ${SERVER_ADDRESS}"
SBATCH_COMMAND="fedavg/run_fold_experiment.sh \
  ${SERVER_CONFIG_PATH} \
  ${EXPERIMENT_DIRECTORY} \
  ${TASK_TYPE} \
  ${LR_VALUE} \
  ${N_CLIENTS} \
  ${SERVER_ADDRESS}"
echo "Running sbatch command ${SBATCH_COMMAND}"
sbatch ${SBATCH_COMMAND}
((SERVER_PORT=SERVER_PORT+1))
done

echo Experiments Launched
