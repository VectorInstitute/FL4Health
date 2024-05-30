#!/bin/bash

#SBATCH --gres=gpu:a40:1
#SBATCH --qos=normal
#SBATCH --mem=64GB
#SBATCH -c 16
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@60
#SBATCH --job-name=nnunet_picai_train
#SBATCH --output=%j_%x.out
#SBATCH --error=%j_%x.err

# Process Inputs
DATASET_NAME=$1
UNET_CONFIG=$2
FOLD=$3
VENV_PATH=$4
PLANS_IDENTIFIER=${5:-'plans'} # Default value of plans string
PRETAINED_WEIGHTS=${6:-''} # Default value of empty string

# Set environment paths that nnUNet expects
export nnUNet_raw="/ssd003/projects/aieng/public/PICAI/nnUNet/nnUNet_raw"
export nnUNet_preprocessed="/ssd003/projects/aieng/public/PICAI/nnUNet/nnUNet_preprocessed"
export nnUNet_results="/ssd003/projects/aieng/public/PICAI/nnUNet/nnUNet_results"

# Log input information
echo "Dataset Name : ${DATASET_NAME}"
echo "Config: ${UNET_CONFIG}"
echo "Dataset Fold: ${FOLD}"
echo "Python Env Path: ${VENV_PATH}"
echo "Plans Identifier: ${PLANS_IDENTIFIER}"

if [[ -z "${PRETRAINED_WEIGHTS}" ]]; then
	echo "Pretrained Weights: ${PRETRAINED_WEIGHTS}"
fi

# Setup before running training
source ${VENV_PATH}bin/activate
BASE_DIR="$nnUNet_results/${DATASET_NAME}/nnUNetTrainer__${PLANS_IDENTIFIER}__${UNET_CONFIG}/"

# Function returning array of args passed to nnUNetv2_train command
get_nnunet_train_args() {
	# $1 Refers to the first argument of this function and not first argument of script
	local checkpoint_bool=$1
	arg_list = ( $DATASET_NAME $UNET_CONFIG $FOLD $PLANS_IDENTIFIER )

	if [[ -z "${PRETRAINED_WEIGHTS}" ]]; then
		arg_list += ${PRETRAINED_WEIGHTS}
	fi
	if [ "$checkpoint_bool" = true ]; then
		arg_list += "--c"
	fi
}

# Function that is called to requeue current slurm job
handler()
{
	echo "Requeue $SLURM_JOB_ID at $(date)"
	scontrol requeue $SLURM_JOB_ID
}

# Function that checks whether to start or resume training, gets train_args
# and runs  training
nnunet_train()
{
	local arg_list = null
	if [ -z "$(ls -A $BASE_DIR)" ]; then
		echo "Start training from scratch as $(date)"
		get_nnunet_train_args false # Sets arg_list
	else
		echo "Start training from checkpoint as $(date)"
		get_nnunet_train_args true # Sets arg_list
	fi
	nnUNetv2_train "${arg_list[@]}"
}

# Trap is used to handle signals or errors that occur during execution
# Here we pass the handler function to initiate a requeue when we get a
# signal that the job will be killed due to time limit 60 seconds prior
trap handler SIGUSR1

# Script must be ran in the background (async) to be able to trap signal
nnunet_train &
wait
