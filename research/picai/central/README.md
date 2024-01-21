# Running Centralized Example 

To train and validate a simple U-Net model on the Preprocessed Dataset described in the [PICAI Documentation](/research/picai/README.md) using a centralized setup, simply submit the `launch.slrm` job to the cluster using:
```
sbatch launch.slrm
```

This script will request compute resources and execute the python file `train.py` to train and validate a U-Net model on a specified fold of the PICAI dataset. The train.py file takes multiple arguments that are currently hardcoded to launch a simple job that will work out of the box for anyone with access to the Vector Cluster. The train.py takes the following parameters: 
- **--base_dir** (str): Base path to the PICAI dataset. Defaults to the current location on the cluster. 
- **--overviews_dir** (str): Path to the directory containing overview files for the train and validation dataset of each split. Defaults to current location on the cluster. 
- **--fold** (int): An integer 0-4 specifiying which cross validation fold to run the experiment. **Required**.
- **--num_channels** (int): The number of input channels. Defaults to 3.
- **--num_classes** (int): The number of output channels. Defaults to 2.
- **--num_epochs** (int): The number of epochs to train and validate the model. Defaults to 5.
- **--checkpoint_dir** (str): The path to store the checkpoint from training. **Required**.
- **--batch_size** (int): The number of samlpes per batch. Defaults to 8.

The arguments for `train.py` used in the `launch.slrm` script can be modified by changing the passed arguments. 
