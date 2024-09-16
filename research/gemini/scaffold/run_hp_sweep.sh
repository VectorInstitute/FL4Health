#!/bin/bash


SERVER_CONFIG_PATH=$1
ARTIFACT_DIR=$2
TASK_TYPE=$3
N_CLIENTS=$4


SERVER_LR=( 0.0001 0.001 0.01 0.1)
CLIENT_LR=( 0.00001 0.0001 0.001 0.01 )


SERVER_PORT=8080

# Create sweep folder
SWEEP_DIRECTORY=""${ARTIFACT_DIR}hp_sweep_results""
echo "Creating sweep folder at ${SWEEP_DIRECTORY}"
mkdir ${SWEEP_DIRECTORY}
echo "Task: ${TASK_TYPE}"

for SERVER_LR_VALUE in "${SERVER_LR[@]}";
do
  for CLIENT_LR_VALUE in "${CLIENT_LR[@]}";
  do
    EXPERIMENT_NAME="ser_${SERVER_LR_VALUE}_lr_${CLIENT_LR_VALUE}"
    echo "Beginning Experiment ${EXPERIMENT_NAME}"
    EXPERIMENT_DIRECTORY="${ARTIFACT_DIR}hp_sweep_results/${EXPERIMENT_NAME}/"
    echo "Creating experiment folder ${EXPERIMENT_DIRECTORY}"
    mkdir "${EXPERIMENT_DIRECTORY}"
    SERVER_ADDRESS="0.0.0.0:${SERVER_PORT}"
    echo "Server Address: ${SERVER_ADDRESS}"
    SBATCH_COMMAND="scaffold/run_fold_experiment.slrm \
      ${SERVER_CONFIG_PATH} \
      ${EXPERIMENT_DIRECTORY} \
      ${TASK_TYPE} \
      ${SERVER_LR_VALUE} \
      ${CLIENT_LR_VALUE} \
      ${N_CLIENTS} \
      ${SERVER_ADDRESS}"
    echo "Running sbatch command ${SBATCH_COMMAND}"
    sbatch ${SBATCH_COMMAND}
    ((SERVER_PORT=SERVER_PORT+1))
  done
done

echo Experiments Launched
