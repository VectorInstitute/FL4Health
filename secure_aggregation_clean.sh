#!/bin/bash

# Define the array of folder paths
folder_paths=(
    "/h/your_usrname_here/FL4Health/logs/"
    # Add more folder paths as needed
)

# Define the extensions to delete
extensions=("*.pth" "*.pkl")

# Iterate through each folder path
for folder_path in "${folder_paths[@]}"; do
    # Iterate through each extension and delete files recursively
    for ext in "${extensions[@]}"; do
        # Delete files with the current extension recursively
        find "$folder_path" -type f -name "$ext" -delete
        echo "Deleted files with extension $ext in folder: $folder_path"
    done
done
