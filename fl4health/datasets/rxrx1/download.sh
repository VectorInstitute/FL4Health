echo "RxRx1 dataset download."
# Define the URL and the target directory and file name
URL="https://storage.googleapis.com/rxrx/rxrx1"
METADATA_URL="https://storage.googleapis.com/rxrx/rxrx1/rxrx1-metadata.zip"
DIRECTORY="/projects/fl4health/datasets/rxrx1_v1.0/"
IMAGE_FILE_NAME="rxrx1-images.zip"
METADATA_FILE="rxrx1-metadata.zip"
IMAGE_FILE_PATH=${DIRECTORY}${IMAGE_FILE_NAME}
METADATA_FILE_PATH=${DIRECTORY}${METADATA_FILE}

# Create the directory if it doesn't exist
mkdir -p "$DIRECTORY"

# Check if the file already exists
if [ -f "$IMAGE_FILE_PATH" ]; then
  echo "File $IMAGE_FILE already exists. No download needed."
else
  echo "Downloading $IMAGE_FILE_NAME"
  wget -O "$IMAGE_FILE_PATH" "$URL/$IMAGE_FILE_NAME"
  if [ $? -eq 0 ]; then
    echo "Download completed successfully."
  else
    echo "Download failed."
  fi
fi

mkdir -p ${DIRECTORY}images/
unzip ${IMAGE_FILE_PATH} -d ${DIRECTORY}images/

# Check if the file already exists
if [ -f "$METADATA_FILE_PATH" ]; then
  echo "File $METADATA_FILE already exists. No download needed."
else
  echo "Downloading $METADATA_FILE"
  wget -O "$METADATA_FILE_PATH" "$URL/$METADATA_FILE"
  if [ $? -eq 0 ]; then
    echo "Download completed successfully."
  else
    echo "Download failed."
  fi
fi

unzip ${METADATA_FILE_PATH} -d ${DIRECTORY}

echo "Download completed."
