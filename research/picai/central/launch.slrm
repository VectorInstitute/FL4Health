#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --error=%j_%x.err
#SBATCH --qos normal
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --job-name=central_picai
#SBATCH --output=%j_%x.out

###############################################
# Usage:
#
# sbatch research/picai/central/launch.slrm folder_for_artifacts / \
#  path_to_desired_env/ fold_id
#
# Example:
# sbatch research/picai/fedavg/launch.slrm \
#   research/fedprox_cluster/artifacts/ \
#   /h/demerson/vector_repositories/fl4health_env/ \
#   0
#
# Notes:
# 1) The sbatch command above should be run from the top level directory of the repository.
# 3) The logging directories need to ALREADY EXIST. The script does not create them.
###############################################

# Note:
#	  ntasks: Total number of processes to use across world
#	  ntasks-per-node: How many processes each node should create

# Process inputs
LOG_DIR=$1
VENV_PATH=$2
FOLD_ID=$3
RUN_NAME=$4

echo "Config Path: ${CONFIG_PATH}"
echo "Python Venv Path: ${VENV_PATH}"
echo "Fold ID: ${FOLD_ID}"

LOG_PATH="${LOG_DIR}client_${JOB_HASH}.log"

echo "Placing logs in: ${LOG_DIR}"
echo "World size: ${SLURM_NTASKS}"
echo "Number of nodes: ${SLURM_NNODES}"
NUM_GPUs=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "GPUs per node: ${NUM_GPUs}"

# Source the environment
source ${VENV_PATH}bin/activate
echo "Active Environment:"
which python

python ~/FL4Health/research/picai/central/train.py --checkpoint_dir ${LOG_DIR} --fold ${FOLD_ID} --run_name ${RUN_NAME} > ${LOG_PATH} 2>&1
