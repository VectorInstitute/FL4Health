echo "PAD-UFES-20 download."
# Define the URL and the target directory and file name
URL="https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/zr7vgbcyr2-1.zip"
DIRECTORY="fl4health/datasets/skin_cancer"
FILE="PAD-UFES-20.zip"
TARGET="$DIRECTORY/$FILE"

# Create the directory if it doesn't exist
mkdir -p "$DIRECTORY"

# Check if the file already exists
if [ -f "$TARGET" ]; then
  echo "File $TARGET already exists. No download needed."
else
  echo "Downloading $FILE to $TARGET..."
  curl -L --progress-bar -o "$TARGET" "$URL"
  if [ $? -eq 0 ]; then
    echo "Download completed successfully."
  else
    echo "Download failed."
  fi
fi

unzip ${DIRECTORY}/PAD-UFES-20.zip -d ${DIRECTORY}/
mkdir ${DIRECTORY}/PAD-UFES-20
mv ${DIRECTORY}/images ${DIRECTORY}/PAD-UFES-20/
mv ${DIRECTORY}/metadata.csv ${DIRECTORY}/PAD-UFES-20/
unzip ${DIRECTORY}/PAD-UFES-20/images/imgs_part_1.zip -d ${DIRECTORY}//PAD-UFES-20/
unzip ${DIRECTORY}/PAD-UFES-20/images/imgs_part_2.zip -d ${DIRECTORY}//PAD-UFES-20/
unzip ${DIRECTORY}/PAD-UFES-20/images/imgs_part_3.zip -d ${DIRECTORY}//PAD-UFES-20/
rm -rf ${DIRECTORY}/PAD-UFES-20/images
mv ${DIRECTORY}/PAD-UFES-20/imgs_part_1/* ${DIRECTORY}/PAD-UFES-20/
mv ${DIRECTORY}/PAD-UFES-20/imgs_part_2/* ${DIRECTORY}/PAD-UFES-20/
mv ${DIRECTORY}/PAD-UFES-20/imgs_part_3/* ${DIRECTORY}/PAD-UFES-20/
rm -rf ${DIRECTORY}/PAD-UFES-20/imgs_part_1
rm -rf ${DIRECTORY}/PAD-UFES-20/imgs_part_2
rm -rf ${DIRECTORY}/PAD-UFES-20/imgs_part_3
rm -rf ${TARGET}

echo "Derm7pt download."

unzip ${DIRECTORY}/release_v0.zip -d ${DIRECTORY}/
mv ${DIRECTORY}/release_v0 ${DIRECTORY}/Derm7pt
rm -rf ${DIRECTORY}/release_v0

echo "HAM10000 download."

mkdir ${DIRECTORY}/HAM10000
mv ${DIRECTORY}/HAM10000_images_part_1.zip ${DIRECTORY}/HAM10000/
mv ${DIRECTORY}/HAM10000_images_part_2.zip ${DIRECTORY}/HAM10000/
unzip ${DIRECTORY}/HAM10000/HAM10000_images_part_1.zip -d ${DIRECTORY}/HAM10000/
unzip ${DIRECTORY}/HAM10000/HAM10000_images_part_2.zip -d ${DIRECTORY}/HAM10000/
rm ${DIRECTORY}/HAM10000/HAM10000_images_part_1.zip
rm ${DIRECTORY}/HAM10000/HAM10000_images_part_2.zip
mv ${DIRECTORY}/HAM10000_metadata ${DIRECTORY}/HAM10000/

# Define the URLs and the target directory
URL_INPUT="https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip"
URL_GROUNDTRUTH="https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_GroundTruth.csv"
URL_METADATA="https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Metadata.csv"

# Download the files using curl
echo "Downloading ISIC_2019_Training_Input.zip..."
curl -L --progress-bar -o "$DIRECTORY/ISIC_2019_Training_Input.zip" "$URL_INPUT"

echo "Downloading ISIC_2019_Training_GroundTruth.csv..."
curl -L --progress-bar -o "$DIRECTORY/ISIC_2019_Training_GroundTruth.csv" "$URL_GROUNDTRUTH"

echo "Downloading ISIC_2019_Training_Metadata.csv..."
curl -L --progress-bar -o "$DIRECTORY/ISIC_2019_Training_Metadata.csv" "$URL_METADATA"

# Unzip the ISIC_2019_Training_Input.zip file
echo "Unzipping ISIC_2019_Training_Input.zip..."
unzip ${DIRECTORY}/ISIC_2019_Training_Input.zip -d ${DIRECTORY}/

# Move the extracted directory and CSV files to ISIC_2019 directory
echo "Organizing files..."
mkdir -p ${DIRECTORY}/ISIC_2019
mv ${DIRECTORY}/ISIC_2019_Training_Input ${DIRECTORY}/ISIC_2019/
mv ${DIRECTORY}/ISIC_2019_Training_GroundTruth.csv ${DIRECTORY}/ISIC_2019/
mv ${DIRECTORY}/ISIC_2019_Training_Metadata.csv ${DIRECTORY}/ISIC_2019/

echo "Cleaning up..."
# Remove the zip file after extraction
rm ${DIRECTORY}/ISIC_2019_Training_Input.zip

echo "Process completed."
