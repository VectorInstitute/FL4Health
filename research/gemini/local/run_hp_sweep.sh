#!/bin/bash


ARTIFACT_DIR=$1
TASK_TYPE=$2
N_CLIENTS=$3
TOTAL_EPOCHS=$4
BATCH_SIZE=$5


LR_VALUES=( 0.00001 0.0001 0.001 0.01 )


# Create sweep folder
SWEEP_DIRECTORY=""${ARTIFACT_DIR}hp_sweep_results""
echo "Creating sweep folder at ${SWEEP_DIRECTORY}"
mkdir ${SWEEP_DIRECTORY}
echo "Task: ${TASK_TYPE}"


for LR_VALUE in "${LR_VALUES[@]}";
do
EXPERIMENT_NAME="lr_${LR_VALUE}"
echo "Beginning Experiment ${EXPERIMENT_NAME}"
EXPERIMENT_DIRECTORY="${ARTIFACT_DIR}hp_sweep_results/${EXPERIMENT_NAME}/"
echo "Creating experiment folder ${EXPERIMENT_DIRECTORY}"
mkdir "${EXPERIMENT_DIRECTORY}"


SBATCH_COMMAND="local/run_fold_experiment.sh \
  ${EXPERIMENT_DIRECTORY} \
  ${TASK_TYPE} \
  ${LR_VALUE} \
  ${N_CLIENTS} \
  ${BATCH_SIZE} \
  ${TOTAL_EPOCHS}"
echo "Running sbatch command ${SBATCH_COMMAND}"
sbatch ${SBATCH_COMMAND}
done

echo Experiments Launched
