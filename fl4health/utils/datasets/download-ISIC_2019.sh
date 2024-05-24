# Define the URLs and the target directory
URL_INPUT="https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip"
URL_GROUNDTRUTH="https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_GroundTruth.csv"
URL_METADATA="https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Metadata.csv"
DIRECTORY="fl4health/utils/datasets"

# Create the directory if it doesn't exist
mkdir -p "$DIRECTORY"

# Download the files using curl
echo "Downloading ISIC_2019_Training_Input.zip..."
curl -L --progress-bar -o "$DIRECTORY/ISIC_2019_Training_Input.zip" "$URL_INPUT"

echo "Downloading ISIC_2019_Training_GroundTruth.csv..."
curl -L --progress-bar -o "$DIRECTORY/ISIC_2019_Training_GroundTruth.csv" "$URL_GROUNDTRUTH"

echo "Downloading ISIC_2019_Training_Metadata.csv..."
curl -L --progress-bar -o "$DIRECTORY/ISIC_2019_Training_Metadata.csv" "$URL_METADATA"

# Define the data path
data_path="fl4health/utils/datasets"

# Unzip the ISIC_2019_Training_Input.zip file
echo "Unzipping ISIC_2019_Training_Input.zip..."
unzip ${data_path}/ISIC_2019_Training_Input.zip -d ${data_path}/

# Move the extracted directory and CSV files to ISIC_2019 directory
echo "Organizing files..."
mkdir -p ${data_path}/ISIC_2019
mv ${data_path}/ISIC_2019_Training_Input ${data_path}/ISIC_2019/
mv ${data_path}/ISIC_2019_Training_GroundTruth.csv ${data_path}/ISIC_2019/
mv ${data_path}/ISIC_2019_Training_Metadata.csv ${data_path}/ISIC_2019/

echo "Cleaning up..."
# Remove the zip file after extraction
rm ${data_path}/ISIC_2019_Training_Input.zip

echo "Process completed."
