#!/bin/bash

###############################################
# Usage:
#
# ./research/flamby/fed_isic2019/fedprox/run_fold_experiment_local.sh \
#   path_to_config.yaml \
#   path_to_folder_for_artifacts/ \
#   path_to_folder_for_dataset/ \
#   path_to_desired_venv/ \
#   mu_value_for_fedprox_loss \
#   client_side_learning_rate_value
#
# Example:
# ./research/flamby/fed_isic2019/fedprox/run_fold_experiment_local.sh \
#   research/flamby/fed_isic2019/fedprox/config.yaml \
#   research/flamby/fed_isic2019/fedprox/hp_results/ \
#   /Users/david/Desktop/FLambyDatasets/fedisic2019/ \
#   0.1 \
#   0.0001
#
# Notes:
# 1) The bash command above should be run from the top level directory of the repository.
# 2) The path_to_folder_for_artifacts must already exist.
# 3) VENV must already be activated when running this script
###############################################

# Process Inputs

SERVER_CONFIG_PATH=$1
ARTIFACT_DIR=$2
DATASET_DIR=$3
CLIENT_MU=$4
CLIENT_LR=$5

# Create the artficat directory
mkdir "${ARTIFACT_DIR}"

RUN_NAMES=( "Run1" "Run2" )

echo "Python Venv Path: ${VENV_PATH}"

echo "Active Environment:"
which python

for RUN_NAME in "${RUN_NAMES[@]}";
do
    # create the run directory
    RUN_DIR="${ARTIFACT_DIR}${RUN_NAME}/"
    mkdir "${RUN_DIR}"

    SERVER_OUTPUT_FILE="${RUN_DIR}server.out"

    # Start the server, divert the outputs to a server file

    echo "Server logging at: ${SERVER_OUTPUT_FILE}"
    echo "Launching Server"

    nohup python -m research.flamby.fed_isic2019.fedprox.server \
        --config_path ${SERVER_CONFIG_PATH} \
        --artifact_dir ${ARTIFACT_DIR} \
        --run_name ${RUN_NAME} \
        > ${SERVER_OUTPUT_FILE} 2>&1 &

    # Sleep for 20 seconds to allow the server to come up.
    sleep 20

    # Start n number of clients and divert the outputs to their own files
    n_clients=6
    for (( c=0; c<${n_clients}; c++ ))
    do
        CLIENT_NAME="client_${c}"
        echo "Launching ${CLIENT_NAME}"

        CLIENT_LOG_PATH="${ARTIFACT_DIR}client_${client_number}.out"
        echo "${CLIENT_NAME} logging at: ${CLIENT_LOG_PATH}"
        nohup python -m research.flamby.fed_isic2019.fedprox.client \
            --artifact_dir ${ARTIFACT_DIR} \
            --dataset_dir ${DATASET_DIR} \
            --run_name ${RUN_NAME} \
            --client_number ${c} \
            --mu ${CLIENT_MU} \
            --learning_rate ${CLIENT_LR} \
            > ${CLIENT_LOG_PATH} 2>&1 &
    done

    echo "FL Processes Running"

    wait

    echo "Finished FL Processes"

done
