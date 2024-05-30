#!/bin/bash

# Define a variable to control the behavior (set to "true" to skip existing directories)
SKIP_EXISTING=true
data_path="fl4health/utils/datasets"

# Create the target directory if it doesn't exist
mkdir -p "$data_path"

# URLs and corresponding output filenames
urls_and_files=(
  "https://repo.ijs.si/hribarr/physionet-challenge-2021-dataset/-/raw/main/WFDB_CPSC2018.tar.gz WFDB_CPSC2018.tar.gz"
  "https://repo.ijs.si/hribarr/physionet-challenge-2021-dataset/-/raw/main/WFDB_CPSC2018_2.tar.gz WFDB_CPSC2018_2.tar.gz"
  "https://repo.ijs.si/hribarr/physionet-challenge-2021-dataset/-/raw/main/WFDB_StPetersburg.tar.gz WFDB_StPetersburg.tar.gz"
  "https://repo.ijs.si/hribarr/physionet-challenge-2021-dataset/-/raw/main/WFDB_PTBXL.tar.gz WFDB_PTBXL.tar.gz"
  "https://repo.ijs.si/hribarr/physionet-challenge-2021-dataset/-/raw/main/WFDB_Ga.tar.gz WFDB_Ga.tar.gz"
  "https://repo.ijs.si/hribarr/physionet-challenge-2021-dataset/-/raw/main/WFDB_ChapmanShaoxing.tar.gz WFDB_ChapmanShaoxing.tar.gz"
  "https://repo.ijs.si/hribarr/physionet-challenge-2021-dataset/-/raw/main/WFDB_Ningbo.tar.gz WFDB_Ningbo.tar.gz"
)

# Function to download and extract files
download_and_extract() {
  local url=$1
  local output=$2

  if [ "$SKIP_EXISTING" = true ] && [ -d "${data_path}/${output%.tar.gz}" ]; then
    echo "Directory ${data_path}/${output%.tar.gz} already exists. Skipping download."
  else
    echo "Downloading $output"
    curl -L -o "${data_path}/$output" "$url"
    echo "Extracting $output"
    tar -xvf "${data_path}/$output" -C "$data_path"
    echo "Removing $output"
    rm "${data_path}/$output"
  fi
}

# Loop through the URLs and corresponding filenames and process them
for url_and_file in "${urls_and_files[@]}"; do
  # Split the string into URL and filename
  url=$(echo $url_and_file | awk '{print $1}')
  file=$(echo $url_and_file | awk '{print $2}')
  download_and_extract "$url" "$file"
done

curl -L -o "${data_path}/weights.csv" "https://github.com/physionetchallenges/evaluation-2021/raw/main/weights.csv"

# Directory containing the .hea files
directory="${data_path}/WFDB_Ga"

# Loop through all .hea files in the directory
for file in "$directory"/*.hea; do
  # Read the first line and the rest of the file separately
  first_line=$(head -n 1 "$file")
  rest_of_file=$(tail -n +2 "$file")

  # Check if the first line contains .mat and modify it if it does
  if [[ "$first_line" == *".mat"* ]]; then
    first_line_modified=${first_line/.mat/}
    # Write the modified first line and the rest of the file back
    echo "$first_line_modified" > "$file"
    echo "$rest_of_file" >> "$file"
  fi
done

echo "adjusted the .hea files for WFDB_Ga to align with the rest"

Run the Python preprocessing script
python fl4health/utils/datasets/preprocess_physionet2021.py \
    "$data_path" \
    --meta-dir "$data_path" \
    --dest "${data_path}/ecg_preprocessed_data" \
    --subset "WFDB_CPSC2018, WFDB_CPSC2018_2, WFDB_Ga, WFDB_PTB, WFDB_PTBXL, WFDB_ChapmanShaoxing, WFDB_Ningbo" \
    --workers 8


mkdir ${data_path}/ecg_manifest

python fl4health/utils/datasets/manifest.py \
    "${data_path}/ecg_preprocessed_data" \
    --subset "ChapmanShaoxing" \
    --combine_subsets "ChapmanShaoxing" \
    --dest "${data_path}/ecg_manifest/ChapmanShaoxing" \
    --valid-percent 0.1

python fl4health/utils/datasets/manifest.py \
    "${data_path}/ecg_preprocessed_data" \
    --subset "CPSC2018, CPSC2018_2" \
    --combine_subsets "CPSC2018, CPSC2018_2" \
    --dest "${data_path}/ecg_manifest/CPSC2018" \
    --valid-percent 0.1

python fl4health/utils/datasets/manifest.py \
    "${data_path}/ecg_preprocessed_data" \
    --subset "Ga" \
    --combine_subsets "Ga" \
    --dest "${data_path}/ecg_manifest/Ga" \
    --valid-percent 0.1

python fl4health/utils/datasets/manifest.py \
    "${data_path}/ecg_preprocessed_data" \
    --subset "Ningbo" \
    --combine_subsets "Ningbo" \
    --dest "${data_path}/ecg_manifest/Ningbo" \
    --valid-percent 0.1

python fl4health/utils/datasets/manifest.py \
    "${data_path}/ecg_preprocessed_data" \
    --subset "PTBXL" \
    --combine_subsets "PTBXL" \
    --dest "${data_path}/ecg_manifest/PTBXL" \
    --valid-percent 0.1

# Define the target directory
REPO_URL="https://github.com/wns823/medical_federated"
REPO_BRANCH="main"
TARGET_DIR="fl4health/utils/datasets/fairseq_signals"
EXTRACT_PATH="medical_federated-main/ecg_federated/fairseq_signals"

# Check if the target directory exists
if [ -d "$TARGET_DIR" ]; then
    echo "Target directory $TARGET_DIR already exists. Doing nothing."
    return 0 2>/dev/null || exit 0
fi

# Create the target directory
mkdir -p "$TARGET_DIR"

# Download the tarball of the repository
echo "Downloading the repository tarball..."
curl -L "$REPO_URL/archive/$REPO_BRANCH.tar.gz" -o repo.tar.gz

# Extract the specific contents from the tarball
echo "Extracting the specific contents from the tarball..."
mkdir temp_extract
tar -xzf repo.tar.gz -C temp_extract

# Move the specific directory to the target directory
mv temp_extract/$EXTRACT_PATH/* "$TARGET_DIR/"

# Clean up
echo "Cleaning up..."
rm -rf repo.tar.gz temp_extract

echo "Done. The contents have been extracted to $TARGET_DIR."
