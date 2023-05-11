SERVER_PORT=$1
SERVER_CONFIG_PATH=$2
LOG_DIR=$3
VENV_PATH=$4

CLIENT_DATA_BASE_PATH = "examples/datasets/mnist_data"
CLIENT_PATH_SUFFIXES=("" "" "" "")

# Start the FL Server and wait until the job starts

sbatch research/run_server.slrm ${SERVER_PORT} ${SERVER_CONFIG_PATH} ${LOG_DIR} ${VENV_PATH}

x=1
while [ $x -le 5 ]
do
  echo "Welcome $x times"
  x=$(( $x + 1 ))
done


for CLIENT_PATH_SUFFIX in ${CLIENT_PATH_SUFFIXES[@]}; do

done
