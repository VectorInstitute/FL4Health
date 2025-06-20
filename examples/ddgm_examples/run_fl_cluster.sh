#!/bin/bash

set -euo pipefail

SERVER_PORT=8081
SERVER_CONFIG_PATH="examples/ddgm_examples/config.yaml"
LOG_DIR="examples/ddgm_examples/log"
VENV_PATH="fl4health_ddgm"

num_clients=500
CLIENT_BATCH_SIZE=20
NUM_SLURM=25 # num of slurm jobs

# num_clients = CLIENT_BATCH_SIZE * NUM_CLIENT_PER_BATCH

if (( NUM_SLURM * CLIENT_BATCH_SIZE != num_clients )); then
  echo "ERROR: num_clients must be a multiple of CLIENT_BATCH_SIZE" >&2
  exit 1
fi

# CLIENT_DATA_BASE_PATH="examples/datasets/mnist_data"
################################
#  Cleanup logic               #
################################
# Arrays to hold every SLURM job ID we submit
declare SERVER_JOB_ID
declare -a CLIENT_JOB_IDS=()

clean() {
  echo
  echo "→ Cleaning up all submitted jobs…"
  if [[ -n "${SERVER_JOB_ID:-}" ]]; then
    echo "   • Cancelling server job ${SERVER_JOB_ID}"
    scancel "${SERVER_JOB_ID}" || true
  fi
  for jid in "${CLIENT_JOB_IDS[@]}"; do
    echo "   • Cancelling client job ${jid}"
    scancel "${jid}" || true
  done

  echo "→ Removing server logs in ${LOG_DIR}"
  rm -f "${LOG_DIR}"/server_log_*.{out,err} 2>/dev/null || true

  echo "→ Removing client logs in ${LOG_DIR}"
  rm -f "${LOG_DIR}"/client_log_*.{out,err} 2>/dev/null || true
  rm -f "${LOG_DIR}"/client_*.out 2>/dev/null || true

  echo "→ Cleanup complete."
}

trap clean SIGINT SIGTERM EXIT


################################
#  Launch the server           #
################################
# Start the FL Server and wait until the job starts
SERVER_JOB_HASH=$(echo $( md5sum <<<$RANDOM ) | head -c 10 )
SERVER_JOB_NAME="fl_server_${SERVER_JOB_HASH}"

SERVER_OUT_LOGS="server_log_${SERVER_JOB_HASH}.out"
SERVER_ERROR_LOGS="server_log_${SERVER_JOB_HASH}.err"

echo "Server Port number: ${SERVER_PORT}"
echo "Config Path: ${SERVER_CONFIG_PATH}"
echo "Log Dir: ${LOG_DIR}"
echo "Server Job Hash: ${SERVER_JOB_HASH}"


SERVER_JOB_ID=$( sbatch --parsable \
    --job-name="${SERVER_JOB_NAME}" \
    --output="${LOG_DIR}/${SERVER_OUT_LOGS}" \
    --error="${LOG_DIR}/${SERVER_ERROR_LOGS}" \
    examples/ddgm_examples/batch_exec/run_server.slrm \
      "${SERVER_PORT}" \
      "${SERVER_CONFIG_PATH}" \
      "${LOG_DIR}" )
echo "→ Server Job ID: ${SERVER_JOB_ID}"
echo


# Wait until the server has started
echo "→ Waiting for server to enter Running state…"
while [[ "$(squeue --noheader -u "${USER}" -j "${SERVER_JOB_ID}" -o %t)" != "R" ]]; do
  sleep 1
done

HOST_NAME=$( scontrol show job "${SERVER_JOB_ID}" \
             | awk -F= '/ NodeList/ {print $2}' )
SERVER_ADDRESS="${HOST_NAME}:${SERVER_PORT}"
echo "→ Server is up at ${SERVER_ADDRESS}"
echo

# HOST_NAME=$(scontrol show job ${JOB_ID} | grep ' NodeList' | awk -F'=' '{ print $2 }' )

# Wait until the server is up and waiting for clients on the requested resources
sleep 10


################################
#  Launch all the clients      #
################################
# Spin up the clients on each disparate node with the server address
echo "→ Launching ${num_clients} clients in batches of ${CLIENT_BATCH_SIZE}"
for i in $(seq 1 "$CLIENT_BATCH_SIZE" "$num_clients"); do

  CLIENT_JOB_HASH=$(md5sum <<<"$RANDOM" | head -c 10)
  CLIENT_JOB_NAME="fl_client_${CLIENT_JOB_HASH}"
  CLIENT_OUT_LOGS="client_log_${CLIENT_JOB_HASH}.out"
  CLIENT_ERROR_LOGS="client_log_${CLIENT_JOB_HASH}.err"

  echo "   • Submitting client ${i} → ${CLIENT_JOB_NAME}"
  jid=$( sbatch --parsable \
      --job-name="${CLIENT_JOB_NAME}" \
      --output="${LOG_DIR}/${CLIENT_OUT_LOGS}" \
      --error="${LOG_DIR}/${CLIENT_ERROR_LOGS}" \
      examples/ddgm_examples/batch_exec/run_client.slrm \
        "${SERVER_ADDRESS}" \
        "${LOG_DIR}" \
        "${i}" \
        "${CLIENT_BATCH_SIZE}" \
        "${CLIENT_JOB_HASH}" )
  echo "     -> Client Job ID: ${jid}"
  CLIENT_JOB_IDS+=( "${jid}" )

done

echo
echo "→ All jobs submitted:"
echo "   Server: ${SERVER_JOB_ID}"
echo "   Clients: ${CLIENT_JOB_IDS[*]}"
echo

read -rp "Press y to terminate all → " yn
if [[ "$yn" =~ ^[Yy]$ ]]; then
  clean
fi

