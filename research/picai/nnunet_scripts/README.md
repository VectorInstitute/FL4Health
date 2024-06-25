## Setting up nnUNet

The nnunetv2 package can be install by including th picai group when installing dependencies to your virtual environment using poetry

```bash
poetry install --with "picai"
```

In order to run the nnUNet autosegmentation pipeline, the following environment variables must be set. Add them to .bashrc for convenience

```bash
export nnUNet_raw="/Path/to/nnUNet_raw/folder"
export nnUNet_preprocessed="/Path/to/nnUNet_preprocessed/folder"
export nnUNet_results="/Path/to/nnUNet_results/folder"
```

Raw datasets must be properly formatted and located in the nnUNet_raw folder. For more detailed information see the following [nnUNet documentation](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md)

## Running nnUNet Scripts
A comprehensive overview of [nnUNet](https://github.com/MIC-DKFZ/nnUNet) that outlines standard and finetuning workflows is available in [nnunet_overview.md](nnunet_overview.md). For convenience, two scripts have been made available: `nnunet_launch.slrm` and `nnunet_launch_fold.slrm`:
`nnunet_launch.slrm` is a slurm script to launch a training job on an already configured experiment (ie planning and preprocessing have occurred). It automatically handles checkpointing and relaunching. Optionally takes a path to a set of pretrained weight to start training from. The script can be invoked as follows:
```
sbatch nnunet_launch.slrm DATASET_NAME UNET_CONFIG FOLD VENV_PATH PLANS_IDENTIFIER PRETRAINED_WEIGHTS
```
`nnunet_launch_fold_experiment.slrm` is a higher-level script that does planning and preprocessing on the dataset then subsequently launches 5 instances of the `nnunet_launch.slrm` script on different training and validation splits. Optionally starts training from a set of pretrained weights. The script can be invoked as follows:
```
sbatch nnunet_launch_fold_experiment.slrm DATASET_NAME UNET_CONFIG VENV_PATH PLANS_IDENTIFIER PRETRAINED_WEIGHTS SOURCE_DATASET_NAME SOURCE_PLANS_IDENTIFIER
```

## Transfer Learning with nnUNet

The transfer_train.py script provides an easy way to automate transfer learning with nnUNet models. Run the script with the `--help` flag for a list of arguments that can be passed to the script.

Transfer train allows taking an nnUNet model that has already been trained on one dataset, and training it further on an additional dataset with the same input and output dimensions. A few assumptions are currently made to make this possible

### Assumptions

- nnUNet must already be properly set up.
- The finetuning dataset must be properly formatted and present in the nnUNet_raw folder
- The plans file for the pretrained model must either be specified by the user or present in the nnUNet_preprocessed under the folder for the pretraining dataset (eg. /Path/nnUNet_preprocessed/Dataset000_pretraining)
- The input and output dimensions of the finetuning dataset must match the input and output dimensions of the pretraining dataset. Although the voxel spacing does not technically need to be the same, having different voxel spacings might affect performance
- The pretrained model must be saved as a pytorch model

For now making these assumptions is not a problem for our use case. Eventually we may want to allow using only certain layers from the pretrained model, allowing datasets with differint input and output dimensions
