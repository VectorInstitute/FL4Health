#!/bin/bash

# This script is used to launch a federated learning experiment on a cluster with each client running on a separate gpu
# The script takes in the following arguments:
# 1. Server Port number
# 2. Server Config Path
# 3. Artifacts Directory
# 4. Server Log Directory
# 5. Client Log Directory
# 6. Python Virtual Environment Path
# An example script would be:

# examples/fedllm_example/training_script/run_fl_cluster.sh 8111 examples/fedllm_example/config.yaml \
# /projects/fl4health/fedllm/artifacts/ /projects/fl4health/fedllm/distributed_logs/server_logs/ \
# /projects/fl4health/fedllm/distributed_logs/client_logs/ \ /projects/fl4health/flower_env/temp_env/


SERVER_PORT=$1
SERVER_CONFIG_PATH=$2
ARTIFACT_DIR=$3
SERVER_LOG_DIR=$4
CLIENT_LOG_DIR=$5
VENV_PATH=$6

echo "Server Port number: ${SERVER_PORT}"
echo "Config Path: ${SERVER_CONFIG_PATH}"
echo "Server Log Dir: ${SERVER_LOG_DIR}"
echo "Client Log Dir: ${CLIENT_LOG_DIR}"
echo "Python Venv Path: ${VENV_PATH}"
echo "Artifacts Dir: ${ARTIFACT_DIR}"



# Spins up 2 clients
n_clients=2

# Start the FL Server and wait until the job starts
SERVER_JOB_HASH=$(echo $( md5sum <<<$RANDOM ) | head -c 10 )
SERVER_JOB_NAME="fl_server_${SERVER_JOB_HASH}"

SERVER_OUT_LOGS="server_log_${SERVER_JOB_HASH}.out"
SERVER_ERROR_LOGS="server_log_${SERVER_JOB_HASH}.err"

echo "Server Port number: ${SERVER_PORT}"
echo "Config Path: ${SERVER_CONFIG_PATH}"
echo "Server Log Dir: ${SERVER_LOG_DIR}"
echo "Client Log Dir: ${CLIENT_LOG_DIR}"
echo "Python Venv Path: ${VENV_PATH}"
echo "Server Job Hash: ${SERVER_JOB_HASH}"

SWEEP_DIRECTORY="${ARTIFACT_DIR}hp_sweep_results"
echo "Creating sweep folder at ${SWEEP_DIRECTORY}"
mkdir ${SWEEP_DIRECTORY}

EXPERIMENT_NAME="test_distributed"
echo "Beginning Experiment ${EXPERIMENT_NAME}"
EXPERIMENT_DIRECTORY="${SWEEP_DIRECTORY}/${EXPERIMENT_NAME}/"
echo "Creating experiment folder ${EXPERIMENT_DIRECTORY}"
mkdir "${EXPERIMENT_DIRECTORY}"

SBATCH_COMMAND="--job-name=${SERVER_JOB_NAME} --output=${SERVER_OUT_LOGS} --error=${SERVER_ERROR_LOGS} \
  examples/fedllm_example/training_script/run_server.slrm \
  ${SERVER_PORT} \
  ${SERVER_CONFIG_PATH} \
  ${EXPERIMENT_DIRECTORY} \
  ${SERVER_LOG_DIR} \
  ${VENV_PATH} \
  ${SERVER_JOB_HASH}"

JOB_ID=$(sbatch --parsable ${SBATCH_COMMAND} )
echo "Server Job ID: ${JOB_ID}"

# Wait until the server has started
SERVER_STATUS="$(squeue --noheader -u ${USER} -j ${JOB_ID} -o %t )"
while [[ $SERVER_STATUS != "R" ]]
do
  sleep 1s
  SERVER_STATUS="$(squeue --noheader -u ${USER} -j ${JOB_ID} -o %t )"
done

HOST_NAME=$(scontrol show job ${JOB_ID} | grep ' NodeList' | awk -F'=' '{ print $2 }' )

SERVER_ADDRESS="${HOST_NAME}:$SERVER_PORT"
echo "Extracted Server Address: ${SERVER_ADDRESS}"

# Wait until the server is up and waiting for clients on the requested resources
sleep 40

# Spin up the clients on each disparate node with the server address

for (( c=0; c<${n_clients}; c++ ))
do


  CLIENT_JOB_HASH=$(echo $( md5sum <<<$RANDOM ) | head -c 10 )
  CLIENT_JOB_NAME="fl_client_${CLIENT_JOB_HASH}"

  echo "Launching ${CLIENT_JOB_NAME}"

  CLIENT_OUT_LOGS="client_log_${CLIENT_JOB_HASH}.out"
  CLIENT_ERROR_LOGS="client_log_${CLIENT_JOB_HASH}.err"

  SBATCH_COMMAND="--job-name=${CLIENT_JOB_NAME} --output=${CLIENT_OUT_LOGS} --error=${CLIENT_ERROR_LOGS} \
    examples/fedllm_example/training_script/run_client_zero.slrm \
    ${SERVER_ADDRESS} \
    ${EXPERIMENT_DIRECTORY} \
    ${CLIENT_LOG_DIR} \
    ${VENV_PATH}\
    ${c} \
    ${CLIENT_JOB_HASH}"

  sbatch ${SBATCH_COMMAND}

  sleep 20
done

echo "Client Jobs Launched"
