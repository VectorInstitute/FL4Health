#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --qos=m

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

# Process inputs
SERVER_PORT=$1
CONFIG_PATH=$2
LOG_DIR=$3
VENV_PATH=$4
N_CLIENTS=$5

# Print the name of the node, this will be the network address
echo "Node Name: ${SLURMD_NODENAME}"
echo "Server Port number: ${SERVER_PORT}"
echo "Config Path: ${CONFIG_PATH}"
echo "Python Venv Path: ${VENV_PATH}"

SERVER_ADDRESS="${SLURMD_NODENAME}:${SERVER_PORT}"

echo "Server Address: ${SERVER_ADDRESS}"

LOG_PATH="${LOG_DIR}server.log"

echo "Placing logs in: ${LOG_DIR}"
echo "World size: ${SLURM_NTASKS}"
echo "Number of nodes: ${SLURM_NNODES}"
NUM_GPUs=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "GPUs per node: ${NUM_GPUs}"

# Source the environment
source ${VENV_PATH}bin/activate
echo "Active Environment"
which python

python -m research.picai.fedavg.server --config_path ${CONFIG_PATH} --server_address ${SERVER_ADDRESS} --artifact_dir ${LOG_DIR} --n_clients ${N_CLIENTS} > ${LOG_PATH} 2>&1

echo ${SERVER_ADDRESS}
