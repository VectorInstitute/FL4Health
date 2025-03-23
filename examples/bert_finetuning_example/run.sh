#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --partition=rtx6000
#SBATCH --qos=normal
#SBATCH --job-name=bert_example
#SBATCH --output=%j_%x.out
#SBATCH --error=%j_%x.err
#SBATCH --time=04:00:00

###############################################
# Usage:
#
# sbatch examples/bert_finetuning_example/run.sh \
#   path_to_config.yaml \
#   path_to_folder_for_artifacts/ \
#   path_to_folder_for_dataset/ \
#   path_to_desired_venv/ \
#   client_learning_rate_value \
#   server_address
#
# Example:
# bash examples/bert_finetuning_example/run.sh \
#   examples/bert_finetuning_example/config.yaml \
#   examples/bert_finetuning_example/ \
#   /examples/datasets/ \
#   /h/ftavakoli/venv/fl_env/ \
#   0.001 \
#   0.0.0.0:8100
#
# Notes:
# 1) The sbatch command above should be run from the top level directory of the repository.
# 2) This example runs the bert fine-tuning example. As such the data paths and python launch commands are hardcoded. If you want to change
# the example you run, you need to explicitly modify the code below.
# 3) The logging directories need to ALREADY EXIST. The script does not create them.
###############################################

# Note:
#	  ntasks: Total number of processes to use across world
#	  ntasks-per-node: How many processes each node should create

# Set NCCL options
# export NCCL_DEBUG=INFO
# NCCL backend to communicate between GPU workers is not provided in vector's cluster.
# Disable this option in slurm.
export NCCL_IB_DISABLE=1

if [[ "${SLURM_JOB_PARTITION}" == "t4v2" ]] || \
    [[ "${SLURM_JOB_PARTITION}" == "rtx6000" ]]; then
    echo export NCCL_SOCKET_IFNAME=bond0 on "${SLURM_JOB_PARTITION}"
    export NCCL_SOCKET_IFNAME=bond0
fi

# Process Inputs

SERVER_CONFIG_PATH=$1
ARTIFACT_DIR=$2
DATASET_DIR=$3
VENV_PATH=$4
CLIENT_LR=$5
SERVER_ADDRESS=$6

NUM_CLIENTS=2


echo "Python Venv Path: ${VENV_PATH}"

echo "World size: ${SLURM_NTASKS}"
echo "Number of nodes: ${SLURM_NNODES}"
NUM_GPUs=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "GPUs per node: ${NUM_GPUs}"

# Source the environment
source ${VENV_PATH}bin/activate
echo "Active Environment:"
which python


SERVER_OUTPUT_FILE="${ARTIFACT_DIR}server.out"

# Start the server, divert the outputs to a server file

echo "Server logging at: ${SERVER_OUTPUT_FILE}"
echo "Launching Server"

nohup python -m examples.bert_finetuning_example.server \
    --config_path ${SERVER_CONFIG_PATH} \
    --server_address ${SERVER_ADDRESS} \
    > ${SERVER_OUTPUT_FILE} 2>&1 &

# Sleep for 20 seconds to allow the server to come up.
sleep 20

# Start n number of clients and divert the outputs to their own files
for (( c=0; c<${NUM_CLIENTS}; c++ ))
do
    CLIENT_NAME="client_${c}"
    echo "Launching ${CLIENT_NAME}"

    CLIENT_LOG_PATH="${ARTIFACT_DIR}${CLIENT_NAME}.out"
    echo "${CLIENT_NAME} logging at: ${CLIENT_LOG_PATH}"
    nohup python -m examples.bert_finetuning_example.client \
        --artifact_dir ${ARTIFACT_DIR} \
        --dataset_path ${DATASET_DIR} \
        --learning_rate ${CLIENT_LR} \
        --server_address ${SERVER_ADDRESS} \
        > ${CLIENT_LOG_PATH} 2>&1 &
done

echo "FL Processes Running"

wait

# Create a file that verifies that the Run concluded properly
touch "${ARTIFACT_DIR}done.out"
echo "Finished FL Processes"
