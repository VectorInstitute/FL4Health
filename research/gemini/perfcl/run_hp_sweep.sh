#!/bin/bash

###############################################
# Usage:
#
#  ./gemini_fl/perfcl/run_hp_sweep.sh \
#   path_to_config.yaml \
#   path_to_folder_for_artifacts/ \
#   task ("mortality")/ \
#   number_clients/ \
#
# Example:
# ./gemini_fl/perfcl/run_hp_sweep.sh \
#   perfcl/config.yaml \
#   perfcl/2_client_results/ \
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


LR_VALUES=( 0.1 0.01 0.001 )
MU_VALUES=( 0.01 0.1 1 5)
GAMMA_VALUES=( 1 5 10 )



SERVER_PORT=8080

# Create sweep folder
SWEEP_DIRECTORY="${ARTIFACT_DIR}hp_sweep_results"
echo "Creating sweep folder at ${SWEEP_DIRECTORY}"
mkdir ${SWEEP_DIRECTORY}
echo "Task: ${TASK_TYPE}"

for LR_VALUE in "${LR_VALUES[@]}"; do
  for MU_VALUE in "${MU_VALUES[@]}"; do
    for GAMMA_VALUE in "${GAMMA_VALUES[@]}"; do

        EXPERIMENT_NAME="gamma_${GAMMA_VALUE}_mu_${MU_VALUE}_lr_${LR_VALUE}"
        echo "Beginning Experiment ${EXPERIMENT_NAME}"
        EXPERIMENT_DIRECTORY="${ARTIFACT_DIR}hp_sweep_results/${EXPERIMENT_NAME}/"
        echo "Creating experiment folder ${EXPERIMENT_DIRECTORY}"
        mkdir "${EXPERIMENT_DIRECTORY}"
        SERVER_ADDRESS="0.0.0.0:${SERVER_PORT}"
        echo "Server Address: ${SERVER_ADDRESS}"
        SBATCH_COMMAND="perfcl/run_fold_experiment.slrm \
          ${SERVER_CONFIG_PATH} \
          ${EXPERIMENT_DIRECTORY} \
          ${TASK_TYPE} \
          ${MU_VALUE} \
          ${LR_VALUE} \
          ${N_CLIENTS} \
          ${SERVER_ADDRESS}\
          ${GAMMA_VALUE}"
        echo "Running sbatch command ${SBATCH_COMMAND}"
        sbatch ${SBATCH_COMMAND}
        ((SERVER_PORT=SERVER_PORT+1))

    done
  done
done

echo Experiments Launched