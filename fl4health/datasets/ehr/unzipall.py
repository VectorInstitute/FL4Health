import os
import gzip
import shutil

def unzip_gz_files(directory):
    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.gz'):
            # Construct full file path
            filepath = os.path.join(directory, filename)
            # Define the output file path
            output_filepath = os.path.join(directory, filename[:-3])
            
            # Unzip the file
            with gzip.open(filepath, 'rb') as f_in:
                with open(output_filepath, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            print(f'Unzipped: {filename}')

# Replace 'your_directory_path' with the path to your directory
unzip_gz_files('fl4health/datasets/ehr/eicu-2.0')