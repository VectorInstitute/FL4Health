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

mkdir -p ${DIRECTORY}/PAD-UFES-20
unzip ${DIRECTORY}/PAD-UFES-20.zip -d ${DIRECTORY}/PAD-UFES-20/
unzip -j ${DIRECTORY}/PAD-UFES-20/images/imgs_part_1.zip -d ${DIRECTORY}/PAD-UFES-20/
unzip -j ${DIRECTORY}/PAD-UFES-20/images/imgs_part_2.zip -d ${DIRECTORY}/PAD-UFES-20/
unzip -j ${DIRECTORY}/PAD-UFES-20/images/imgs_part_3.zip -d ${DIRECTORY}/PAD-UFES-20/
rm -r ${DIRECTORY}/PAD-UFES-20/images

if [ -n "${TARGET}" ]; then
    rm -r ${TARGET}
else
    echo "TARGET is empty. Skipping rm -r to avoid accidental deletion."
fi


echo "Derm7pt unzipping. Reminder: Download release_v0.zip from SFU: https://derm.cs.sfu.ca/Welcome.html\
  and place it under ${DIRECTORY}."

mkdir -p ${DIRECTORY}/Derm7pt
unzip -j ${DIRECTORY}/release_v0.zip -d ${DIRECTORY}/Derm7pt
rm ${DIRECTORY}/release_v0.zip

echo "HAM10000 unzipping. Reminder: Download HAM10000_images_part_1.zip, HAM10000_images_part_2.zip,\
 and HAM10000_metadata.tab from Harvard Dataverse:\
  https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T \
  and place them under ${DIRECTORY}."

mkdir -p ${DIRECTORY}/HAM10000
unzip ${DIRECTORY}/HAM10000_images_part_1.zip -d ${DIRECTORY}/HAM10000/
unzip ${DIRECTORY}/HAM10000_images_part_2.zip -d ${DIRECTORY}/HAM10000/
rm ${DIRECTORY}/HAM10000_images_part_1.zip
rm ${DIRECTORY}/HAM10000_images_part_2.zip
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

echo "Creating ISIC_2019 directory..."
mkdir -p ${DIRECTORY}/ISIC_2019
echo "Unzipping ISIC_2019_Training_Input.zip into ISIC_2019 directory..."
unzip ${DIRECTORY}/ISIC_2019_Training_Input.zip -d ${DIRECTORY}/ISIC_2019

echo "Cleaning up..."
# Remove the zip file after extraction
rm ${DIRECTORY}/ISIC_2019_Training_Input.zip

echo "Process completed, running preprocess."

python -m fl4health.datasets.skin_cancer.preprocess_skin

echo "Preprocess completed."
