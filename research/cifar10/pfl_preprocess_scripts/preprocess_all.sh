#!/bin/bash

VENV_PATH=$1

SEEDS=( 2024 2025 2026 )
BETAS=( 0.1 0.5 5.0 )
NUM_PARTITIONS=( 7 7 7 )

ORIGINAL_DATA_DIR="research/cifar10/datasets/cifar10/"
DESTINATION_DIRS=( \
    "research/cifar10/datasets/cifar10/" \
    "research/cifar10/datasets/cifar10/" \
    "research/cifar10/datasets/cifar10/" \
    )

echo "Python Venv Path: ${VENV_PATH}"

for index in "${!DESTINATION_DIRS[@]}";
do

  echo "Preprocessing CIFAR with SEED: ${SEEDS[index]} and BETA: ${BETAS[index]}"
  echo "Number of partitions: ${NUM_PARTITIONS[index]}"
  echo "Destination of partitions: ${DESTINATION_DIRS[index]}"

  CLIENT_OUT_LOGS="cifar_preprocess_log_${SEEDS}_${BETAS}_${NUM_PARTITIONS}.out"
  CLIENT_ERROR_LOGS="cifar_preprocess_log_${SEEDS}_${BETAS}_${NUM_PARTITIONS}.err"

  SBATCH_COMMAND="--job-name=cifar_preprocess_${BETA} --output=${CLIENT_OUT_LOGS} --error=${CLIENT_ERROR_LOGS} \
    research/cifar10/pfl_preprocess_scripts/preprocess.slrm \
    ${VENV_PATH} ${ORIGINAL_DATA_DIR} ${DESTINATION_DIRS[index]} ${SEEDS[index]} ${BETAS[index]} \
    ${NUM_PARTITIONS[index]} ${DESTINATION_DIRS[index]}" \

  sbatch ${SBATCH_COMMAND}
done

echo "Preprocess Jobs Launched"
