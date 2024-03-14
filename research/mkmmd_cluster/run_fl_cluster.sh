#!/bin/bash

SERVER_PORT=$1
SERVER_CONFIG_PATH=$2
ARTIFACT_DIR=$3
DATASET_DIR=$4
SERVER_LOG_DIR=$5
CLIENT_LOG_DIR=$6
VENV_PATH=$7

echo "Server Port number: ${SERVER_PORT}"
echo "Config Path: ${SERVER_CONFIG_PATH}"
echo "Server Log Dir: ${SERVER_LOG_DIR}"
echo "Client Log Dir: ${CLIENT_LOG_DIR}"
echo "Python Venv Path: ${VENV_PATH}"
echo "Artifacts Dir: ${ARTIFACT_DIR}"
echo "Dataset Dir: ${DATASET_DIR}"


# Spins up 3 clients
n_clients=3

LR_VALUE=( 0.001 )
MU_VALUE=( 0.1 )
GAMMA_VALUE=( 0.1 )

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

SWEEP_DIRECTORY="${ARTIFACT_DIR}hp_sweep_results_mkmmd_dist"
echo "Creating sweep folder at ${SWEEP_DIRECTORY}"
mkdir ${SWEEP_DIRECTORY}

EXPERIMENT_NAME="lr_${LR_VALUE}_mu_${MU_VALUE}_gamma_${GAMMA_VALUE}"
echo "Beginning Experiment ${EXPERIMENT_NAME}"
EXPERIMENT_DIRECTORY="${SWEEP_DIRECTORY}/${EXPERIMENT_NAME}/"
echo "Creating experiment folder ${EXPERIMENT_DIRECTORY}"
mkdir "${EXPERIMENT_DIRECTORY}"

SBATCH_COMMAND="--job-name=${SERVER_JOB_NAME} --output=${SERVER_OUT_LOGS} --error=${SERVER_ERROR_LOGS} \
  research/mkmmd_cluster/run_server.slrm \
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
sleep 20

# Spin up the clients on each disparate node with the server address

for (( c=0; c<${n_clients}; c++ ))
do


  CLIENT_JOB_HASH=$(echo $( md5sum <<<$RANDOM ) | head -c 10 )
  CLIENT_JOB_NAME="fl_client_${CLIENT_JOB_HASH}"

  echo "Launching ${CLIENT_JOB_NAME}"

  CLIENT_OUT_LOGS="client_log_${CLIENT_JOB_HASH}.out"
  CLIENT_ERROR_LOGS="client_log_${CLIENT_JOB_HASH}.err"

  SBATCH_COMMAND="--job-name=${CLIENT_JOB_NAME} --output=${CLIENT_OUT_LOGS} --error=${CLIENT_ERROR_LOGS} \
    research/mkmmd_cluster/run_client.slrm \
    ${SERVER_ADDRESS} \
    ${EXPERIMENT_DIRECTORY} \
    ${DATASET_DIR} \
    ${CLIENT_LOG_DIR} \
    ${VENV_PATH}\
    ${LR_VALUE} \
    ${MU_VALUE} \
    ${GAMMA_VALUE} \
    ${c} \
    ${CLIENT_JOB_HASH}"

  sbatch ${SBATCH_COMMAND}
done

echo "Client Jobs Launched"
