#!/bin/bash

###############################################
# Usage:
#
#  ./research/cifar10/adaptive_pfl/ditto/run_hp_sweep.sh \
#   path_to_config.yaml \
#   path_to_folder_for_artifacts/ \
#   path_to_folder_for_dataset/ \
#   path_to_desired_venv/
#
# Example:
# ./research/cifar10/adaptive_pfl/ditto/run_hp_sweep.sh \
#   research/cifar10/adaptive_pfl/ditto/config.yaml \
#   research/cifar10/adaptive_pfl/ditto \
#   /datasets/cifar10 \
#   /h/demerson/vector_repositories/fl4health_env/
#
# Notes:
# 1) The bash command above should be run from the top level directory of the repository.
###############################################

SERVER_CONFIG_PATH=$1
ARTIFACT_DIR=$2
DATASET_DIR=$3
VENV_PATH=$4

LR_VALUES=( 0.0001 0.001 0.01 0.1 )
# Note: These values must correspond to values for the preprocessed CIFAR datasets
BETA_VALUES=( 0.1 0.5 5.0 )
LAM_VALUES=( 0.001 1.0 )
ADAPT_BOOL=( "TRUE" "FALSE" )

SERVER_PORT=8100

# Create sweep folder
SWEEP_DIRECTORY="${ARTIFACT_DIR}hp_sweep_results"
echo "Creating sweep folder at ${SWEEP_DIRECTORY}"
mkdir ${SWEEP_DIRECTORY}

for BETA_VALUE in "${BETA_VALUES[@]}"; do
  echo "Creating folder for beta ${BETA_VALUE}"
  mkdir "${SWEEP_DIRECTORY}/beta_${BETA_VALUE}"
  for LR_VALUE in "${LR_VALUES[@]}";
  do
    for LAM_VALUE in "${LAM_VALUES[@]}";
    do
      for ADAPT in "${ADAPT_BOOL[@]}";
      do
        EXPERIMENT_NAME="lr_${LR_VALUE}_beta_${BETA_VALUE}_lam_${LAM_VALUE}"
        if [[ ${ADAPT} == "TRUE" ]]; then
          EXPERIMENT_NAME="${EXPERIMENT_NAME}_adapt"
        fi
        echo "Beginning Experiment ${EXPERIMENT_NAME}"
        EXPERIMENT_DIRECTORY="${SWEEP_DIRECTORY}/beta_${BETA_VALUE}/${EXPERIMENT_NAME}/"
        echo "Creating experiment folder ${EXPERIMENT_DIRECTORY}"
        mkdir "${EXPERIMENT_DIRECTORY}"
        SERVER_ADDRESS="0.0.0.0:${SERVER_PORT}"
        echo "Server Address: ${SERVER_ADDRESS}"
        SBATCH_COMMAND="research/cifar10/adaptive_pfl/ditto/run_fold_experiment.slrm \
          ${SERVER_CONFIG_PATH} \
          ${EXPERIMENT_DIRECTORY} \
          ${DATASET_DIR} \
          ${VENV_PATH} \
          ${LR_VALUE} \
          ${LAM_VALUE} \
          ${SERVER_ADDRESS} \
          ${BETA_VALUE} \
          ${ADAPT}"
        sbatch ${SBATCH_COMMAND}
        ((SERVER_PORT=SERVER_PORT+1))
      done
    done
  done
done
echo Experiments Launched
