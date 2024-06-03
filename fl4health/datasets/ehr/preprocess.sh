DATA_STORAGE_PATH="fl4health/datasets/ehr"
EICU_PATH="$DATA_STORAGE_PATH/eicu-2.0"
BENCHMARK_PATH="$EICU_PATH/federated_preprocessed_data/final_datasets_for_sharing"

# Function to check if a directory exists
check_directory() {
    if [ ! -d "$1" ]; then
        echo "Error: Directory $1 does not exist."
        return 1
    else
        return 0
    fi
}

# Check if data_storage directory exists
check_directory "$DATA_STORAGE_PATH"
DATA_STORAGE_STATUS=$?

if [ $DATA_STORAGE_STATUS -ne 0 ]; then
    echo "Please create the data storage directory using the following command:"
    echo "mkdir -p $DATA_STORAGE_PATH"
fi

# Check if eICU directory exists
check_directory "$EICU_PATH"
EICU_STATUS=$?

if [ $EICU_STATUS -ne 0 ]; then
    echo "Please download the eICU database from the following link and place it in $DATA_STORAGE_PATH:"
    echo "https://eicu-crd.mit.edu/"
fi

# Check if benchmark dataset directory exists
check_directory "$BENCHMARK_PATH"
BENCHMARK_STATUS=$?

if [ $BENCHMARK_STATUS -ne 0 ]; then
    echo "Please download the benchmark dataset from the following repository and place it in $EICU_PATH/federated_preprocessed_data:"
    echo "https://github.com/mmcdermott/comprehensive_MTL_EHR"
fi

# Run the Python preprocessing script if all directories are present
if [ $DATA_STORAGE_STATUS -eq 0 ] && [ $EICU_STATUS -eq 0 ] && [ $BENCHMARK_STATUS -eq 0 ]; then
    echo "All necessary directories are present. Running the preprocessing script (this may take a while) ..."
    python fl4health/datasets/ehr/preprocess.py --data_path "$DATA_STORAGE_PATH"
else
    echo "Please ensure all necessary directories are present before running the script."
fi
