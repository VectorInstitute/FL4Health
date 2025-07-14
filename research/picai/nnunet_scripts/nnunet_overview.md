## Intro to nnUNet
[nnUNet](https://github.com/MIC-DKFZ/nnUNet) is an experiment configuration pipeline that automatically configures a segmentation model and associated training procedure based on the characteristics of a given medical dataset and available compute. Empirically, nnUNet demonstrates strong performance on a wide range of medical segmentation tasks across modalities such as MRI, CT and others. This document serves as a brief introduction to nnUNet as it relates to FL4Health and the PICAI dataset. For more information about nnUNet, check out the extensive [documentation](https://github.com/MIC-DKFZ/nnUNet/tree/master/documentation).

### Setting Environment Variables
nnUNet expects that three environmental variables have been set with their corresponding paths: `nnUNet_raw`, `nnUNet_preprocessed` and `nnUNet_results`. These paths must be set before a user can run any of the scripts. On the Vector cluster, these paths are as follows:
```
nnUNet_raw="/ssd003/projects/aieng/public/PICAI/nnUNet/nnUNet_raw"
nnUNet_preprocessed="/ssd003/projects/aieng/public/PICAI/nnUNet/nnUNet_preprocessed"
nnUNet_results="/ssd003/projects/aieng/public/PICAI/nnUNet/nnUNet_results"
```
By default, these paths are automatically set in the `nnunet_launch.slrm` script. Simply, change the paths in the script if you would like to use something different. For information about setting the environment variables, visit the [nnUNet Environment Variables Documentation](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/set_environment_variables.md)

### Dataset Formatting
Datasets must be located in the nnUNet_raw folder. Each segmentation dataset is stored as a separate 'Dataset'. Datasets are associated with a dataset ID, a three digit integer, and a dataset name (which you can freely choose): For example, Dataset005_Prostate has 'Prostate' as dataset name and the dataset id is 5. Datasets are stored in the nnUNet_raw folder like this:
```
nnUNet_raw/
├── Dataset001_BrainTumour
├── Dataset002_Heart
├── Dataset003_Liver
├── Dataset004_Hippocampus
├── Dataset005_Prostate
├── ...
```
Within each dataset folder, the following structure is expected:
```
Dataset001_BrainTumour/
├── dataset.json
├── imagesTr
└── labelsTr
```

The expected files in each directory are defined as follows:
- **imagesTr**: Images belonging to the training cases.
- **labelsTr**: Ground truth segmentation maps for the training cases.
- **dataset.json**: Metadata about the dataset.

For information about the dataset formatting, visit the [nnUNet Dataset Formatting Documentation](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md).

### Extracting Configuration, Preprocessing and Training
In order to extract the configuration and run preprocessing, run the following command:

```
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
```
nnUNetv2_plan_and_preprocess will create a new subfolder in your nnUNet_preprocessed folder named after the dataset. Once the command is completed there will be a dataset_fingerprint.json file as well as a nnUNetPlans.json file for you to look at (in case you are interested!). There will also be subfolders containing the preprocessed data for your UNet configurations.

The next step is to train the configured model on the preprocessed dataset:
```
nnUNetv2_train DATASET_NAME_OR_ID UNET_CONFIGURATION FOLD [additional options, see -h]
```

For information about the dataset formatting, visit the [nnUNet Main Documentation](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how_to_use_nnunet.md).


## FineTuning nnUNet
### Intro
Many training hyperparameters such as patch size and network topology differ between datasets as a result of the automated dataset analysis and experiment planning nnU-Net is known for. So, out of the box, it is not possible to simply take the network weights from some dataset and then reuse them for another.

Consequently, the plans need to be aligned between the two tasks. In this README we outline how to finetune a model (that has already been trained on a source dataset) on target dataset.

### Terminology
- **Source Dataset**: The dataset which the model has already been trained on.
- **Target Dataset**: The dataset that we wish to finetune the model on.

### Pretraining on the Source Dataset (Optional)
If you have not already pretrained on the source dataset, proceed with standard nnUNet training procedure. The first step in this process is to extract the plan from the source dataset and preprocess it accordingly:
```
nnUNetv2_plan_and_preprocess -d SOURCE_DATASET
```
Next, train the model on the source dataset:
```
nnUNetv2_train SOURCE_DATASET UNET_CONFIGURATION FOLD
```
where UNET_CONFIGURATION is a string that identifies the requested U-Net configuration (defaults: 2d, 3d_fullres, 3d_lowres, 3d_cascade_lowres). FOLD specifies which fold of the 5-fold-cross-validation is trained.

### Transferring Plan from Source to Target Dataset
Assuming the model has already been trained on a source dataset, we can proceed  with transferring the plan from source to the target dataset. First, if it's not yet available, extract the fingerprint of the target dataset:
```
nnUNetv2_extract_fingerprint -d TARGET_DATASET
```
Then the plan can be transferred from the source dataset to the target dataset:
```
nnUNetv2_move_plans_between_datasets -s SOURCE_DATASET -t TARGET_DATASET -sp SOURCE_PLANS_IDENTIFIER -tp TARGET_PLANS_IDENTIFIER
```
`SOURCE_PLANS_IDENTIFIER` is probably nnUNetPlans (or simply plans) unless you changed the experiment planner in nnUNetv2_plan_and_preprocess. For `TARGET_PLANS_IDENTIFIER` we recommend you set something custom in order to not overwrite default plans.

**Note:** EVERYTHING is transferred between the datasets. Not just the network topology, batch size and patch size but also the normalization scheme!

### Preprocessing Target Dataset and FineTuning
Now that the plan has been transferred, the target dataset can be preprocessed:
```
nnUNetv2_preprocess -d TARGET_DATASET -plans_name TARGET_PLANS_IDENTIFIER
```
Then we can proceed with finetuning on the target dataset:
```
nnUNetv2_train TARGET_DATASET CONFIG FOLD -pretrained_weights PATH_TO_CHECKPOINT
```
where `PATH_TO_CHECKPOINT` is the path to the model weights of the pretrained model.
