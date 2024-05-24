# Define the URL and the target directory and file name
URL="https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/zr7vgbcyr2-1.zip"
DIRECTORY="fl4health/utils/datasets"
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
