#!/bin/bash

###############################################
# Usage:
#
# ./research/flamby/fed_isic2019/fedprox/run_hp_sweep_test.sh \
#   path_to_config.yaml \
#   path_to_folder_for_artifacts/ \
#   path_to_folder_for_dataset/ \
#
# Example:
# ./research/flamby/fed_isic2019/fedprox/run_hp_sweep_test.sh \
#   research/flamby/fed_isic2019/fedprox/config.yaml \
#   research/flamby/fed_isic2019/fedprox/ \
#   /Users/david/Desktop/FLambyDatasets/fedisic2019/
#
# Notes:
# 1) The bash command above should be run from the top level directory of the repository.
# 2) VENV should already be activated when running bash
###############################################

SERVER_CONFIG_PATH=$1
ARTIFACT_DIR=$2
DATASET_DIR=$3

MU_VALUES=( 0.01 0.1 )
LR_VALUES=( 0.00001 0.0001 )

# Create sweep folder
SWEEP_DIRECTORY="${ARTIFACT_DIR}hp_sweep_results"
echo "Creating sweep folder at ${SWEEP_DIRECTORY}"
mkdir ${SWEEP_DIRECTORY}

for MU_VALUE in "${MU_VALUES[@]}";
do
  for LR_VALUE in "${LR_VALUES[@]}";
  do
    EXPERIMENT_NAME="mu_${MU_VALUE}_lr_${LR_VALUE}"
    echo "Beginning Experiment ${EXPERIMENT_NAME}"
    EXPERIMENT_DIRECTORY="${SWEEP_DIRECTORY}/${EXPERIMENT_NAME}/"
    echo "Creating experiment folder ${EXPERIMENT_DIRECTORY}"
    mkdir "${EXPERIMENT_DIRECTORY}"
    ./research/flamby/fed_isic2019/fedprox/run_fold_experiment_test.sh \
      ${SERVER_CONFIG_PATH} \
      ${EXPERIMENT_DIRECTORY} \
      ${DATASET_DIR} \
      ${MU_VALUE} \
      ${LR_VALUE}

    wait
  done
done

echo Experiments Concluded
