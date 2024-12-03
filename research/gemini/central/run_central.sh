#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5120MB
#SBATCH --partition=gpu
#SBATCH --qos=hipri
#SBATCH --job-name=0.00001
#SBATCH --output=0.00001.out
#SBATCH --error=0.00001.err
#SBATCH --mail-user=your_email@vectorinstitute.ai



# Process Inputs
ARTIFACT_DIR=$1
TASK_TYPE=$2
LEARNING_RATE=$3
BATCH_SIZE=$4
NUM_EPOCHS=$5

echo "Task type: ${TASK_TYPE}"

# Source the environment
. ~/py39/bin/activate
echo "Active Environment:"
which python

RUN_NAMES=( "Run1" "Run2" "Run3" "Run4" "Run5" )


# Create sweep folder
SWEEP_DIRECTORY=""${ARTIFACT_DIR}hp_sweep_results""
echo "Creating sweep folder at ${SWEEP_DIRECTORY}"
mkdir ${SWEEP_DIRECTORY}


EXPERIMENT_NAME="lr_${LEARNING_RATE}_epochs_${NUM_EPOCHS}"
echo "Beginning Experiment ${EXPERIMENT_NAME}"
EXPERIMENT_DIRECTORY="${ARTIFACT_DIR}hp_sweep_results/${EXPERIMENT_NAME}/"
mkdir "${EXPERIMENT_DIRECTORY}"

for RUN_NAME in "${RUN_NAMES[@]}";
do
    # create the run directory
    RUN_DIR="${EXPERIMENT_DIRECTORY}${RUN_NAME}/"
    echo "Starting Run and logging artifacts at ${RUN_DIR}"

    if [ -d "${RUN_DIR}" ]
    then
        echo "Run did not finished correctly. Re-running."
        rm -r "${RUN_DIR}"
        mkdir "${RUN_DIR}"

    else
        # Directory doesn't exist yet, so we create it.
        echo "Run directory does not exist. Creating it."
        mkdir "${RUN_DIR}"
    fi


    OUTPUT_FILE="${RUN_DIR}output.out"
    echo "Server logging at: ${OUTPUT_FILE}"

# RUN_NAME="Run1"
# RUN_DIR="${EXPERIMENT_DIRECTORY}${RUN_NAME}/"
# echo "Starting Run and logging artifacts at ${RUN_DIR}"
# mkdir "${RUN_DIR}"

# OUTPUT_FILE="${RUN_DIR}output.out"


    python -m central.train --artifact_dir ${EXPERIMENT_DIRECTORY} --run_name ${RUN_NAME} --learning_rate ${LEARNING_RATE} --task ${TASK_TYPE} --batch_size ${BATCH_SIZE} --num_epochs ${NUM_EPOCHS}

done
echo Experiments Launched
