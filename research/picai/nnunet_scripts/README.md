## Running nnUNet Scripts
A comprehensive overview of the [nnUNet](https://github.com/MIC-DKFZ/nnUNet) that outlines standard and finetuning workflows is available in [nnunet_overview.md](nnunet_overview.md). For convenience, two scripts have been made available: `nnunet_launch.slrm` and `nnunet_launch_fold.slrm`:
`nnunet_launch.slrm` is a slurm script to launch a training job on an already configured experiment (ie planning and preprocessing have occurred). It automatically handles checkpointing and relaunching. Optionally takes a path to set of pretrained weight to start training from. The script can be invoked as follows:
```
sbatch nnunet_launch.slrm DATASET_NAME UNET_CONFIG FOLD VENV_PATH PLANS_IDENTIFIER PRETRAINED_WEIGHTS
```
`nnunet_launch_fold_experiment.slrm` is a higher-level script that does planning and preprocessing on the dataset then subsequently launches 5 instances of the `nnunet_launch.slrm` script on different training and validation splits. Optionally starts training from a set of pretrained weights. The script can be invoked as follows:
```
sbatch nnunet_launch_fold_experiment.slrm DATASET_NAME UNET_CONFIG VENV_PATH PLANS_IDENTIFIER PRETRAINED_WEIGHTS SOURCE_DATASET_NAME SOURCE_PLANS_IDENTIFIER
```
