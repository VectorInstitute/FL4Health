#!/bin/bash

###############################################
# Usage:
#
#  ./research/flamby/fed_isic2019/moon_mkmmd/run_hp_sweep.sh \
#   path_to_config.yaml \
#   path_to_folder_for_artifacts/ \
#   path_to_folder_for_dataset/ \
#   path_to_desired_venv/
#
# Example:
# ./research/flamby/fed_isic2019/moon_mkmmd/run_hp_sweep.sh \
#   research/flamby/fed_isic2019/moon_mkmmd/config.yaml \
#   research/flamby/fed_isic2019/moon_mkmmd/ \
#   /Users/david/Desktop/FLambyDatasets/fedisic2019/ \
#   /h/demerson/vector_repositories/fl4health_env/
#
# Notes:
# 1) The bash command above should be run from the top level directory of the repository.
###############################################

SERVER_CONFIG_PATH=$1
ARTIFACT_DIR=$2
DATASET_DIR=$3
VENV_PATH=$4

# FedISIC LR Hyperparmeters from paper are not suitable for AdamW
LR_VALUES=( 0.00001 0.0001 0.001 0.01 0.1 )
MU_VALUES=( 0 0.1 1 10 )
GAMMA_VALUES=( 0 0.1 1 10 )
L2_VALUES=( 0.01 0.1 1.0 )

SERVER_PORT=8100

# Create sweep folder
SWEEP_DIRECTORY="${ARTIFACT_DIR}hp_sweep_results"
echo "Creating sweep folder at ${SWEEP_DIRECTORY}"
mkdir ${SWEEP_DIRECTORY}

for LR_VALUE in "${LR_VALUES[@]}";do
  for MU_VALUE in "${MU_VALUES[@]}";do
    for GAMMA_VALUE in "${GAMMA_VALUES[@]}";do
      for L2_VALUE in "${L2_VALUES[@]}";do
        EXPERIMENT_NAME="lr_${LR_VALUE}_mu_${MU_VALUE}_gamma_${GAMMA_VALUE}_l2_${L2_VALUE}"
        echo "Beginning Experiment ${EXPERIMENT_NAME}"
        EXPERIMENT_DIRECTORY="${SWEEP_DIRECTORY}/${EXPERIMENT_NAME}/"
        echo "Creating experiment folder ${EXPERIMENT_DIRECTORY}"
        mkdir "${EXPERIMENT_DIRECTORY}"
        SERVER_ADDRESS="0.0.0.0:${SERVER_PORT}"
        echo "Server Address: ${SERVER_ADDRESS}"
        SBATCH_COMMAND="research/flamby/fed_isic2019/moon_mkmmd/run_fold_experiment.slrm \
          ${SERVER_CONFIG_PATH} \
          ${EXPERIMENT_DIRECTORY} \
          ${DATASET_DIR} \
          ${VENV_PATH} \
          ${LR_VALUE} \
          ${SERVER_ADDRESS} \
          ${MU_VALUE} \
          ${GAMMA_VALUE} \
          ${L2_VALUE}"
        echo "Running sbatch command ${SBATCH_COMMAND}"
        sbatch ${SBATCH_COMMAND}
        ((SERVER_PORT=SERVER_PORT+1))
      done
    done
  done
done

echo Experiments Launched