## Setting up nnUNet

The nnunetv2 package can be installed by including the picai extra when installing dependencies to your virtual environment using uv

```bash
uv sync --extra picai
```

In order to run the nnUNet autosegmentation pipeline, the following environment variables must be set. Add them to .bashrc for convenience

```bash
export nnUNet_raw="/Path/to/nnUNet_raw/folder"
export nnUNet_preprocessed="/Path/to/nnUNet_preprocessed/folder"
export nnUNet_results="/Path/to/nnUNet_results/folder"
```

Raw datasets must be properly formatted and located in the nnUNet_raw folder. For more detailed information see the following [nnUNet documentation](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md)

## Running nnUNet Slurm Scripts
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

The transfer_train.py script provides an easy way to automate transfer learning with nnUNet models. Run the script with the `--help` flag for a list of arguments that can be passed to the script. Below is an example invocation with the required flags (and the trainer flag to limit the number of epochs, the default for nnunet is 1000)

```bash
python transfer_train.py --finetune_id 012 --pretrain_id 011 --configs 2d 3d_fullres --trainer nnUNetTrainer_5epochs --pretrain_checkpoints /path/to/2d/checkpoint.pth /path/to/3d_fullres/checkpoint.pth
```

Transfer train allows taking an nnUNet model that has already been trained on one dataset, and training it further on an additional dataset with the same input and output dimensions. A few assumptions are currently made to make this possible

### Assumptions

- nnUNet must already be properly set up.
- The finetuning dataset must be properly formatted and present in the nnUNet_raw folder
- The plans file for the pretrained model must either be specified by the user or present in the nnUNet_preprocessed under the folder for the pretraining dataset (eg. /Path/nnUNet_preprocessed/Dataset000_pretraining)
- The input and output dimensions of the finetuning dataset must match the input and output dimensions of the pretraining dataset. Although the voxel spacing does not technically need to be the same, having different voxel spacings might affect performance
- The pretrained model must be saved as a pytorch model

For now making these assumptions is not a problem for our use case. Eventually we may want to allow using only certain layers from the pretrained model, allowing datasets with different input and output dimensions

## Full Inference and Evaluation Pipeline

The predict_and_eval.py script provides an all in one pipeline to run inference,
lesion detection and model evaluation. For a more detailed description run
```python predict_and_eval.py -h```. The required arguments are a config file, the input data and the ground truth labels. An example invocation is shown below

```bash
python predict_and_eval.py --config-path cktp.yaml --input-folder path/to/input/data/folder --labels-folder path/to/label/folder
```

One must first however create a yaml config file that provides the necessary information. The config must contain paths to a dataset json, the plans file and nnunet model checkpoints. An example config file is shown below

```yaml
dataset_json: /Path/to/a/dataset.json
plans: /Path/to/plans.json
2d:
  - /Path/to/a/2d/model/checkpoint.pth
  - /Path/to/a/2d/nnunet/results/folder
3d_fullres:
  - /Path/to/a/3d_fullres/model/checkpoint.pth
```

## Only Inference

The predict.py script can be used to do inference on new data using nnunet models. The data on which the model is predicting must be formatted as an nnUNet_raw
dataset. Several model checkpoints from different folds or entirely different
configs can be ensembled using this script. Run ```python predict.py -h``` for more information

An example invocation of predict.py is shown below

```bash
python predict.py --config-path path/to/config.yaml --input-folder path/to/input/data/folder
```

## Only Evaluation

The PICAI competition from which the picai datasets originates used the following metric to score models

$$PICAI\ Score=\frac{AUROC+AP}{2}
$$

Where AUROC is the Area Under the Receiver Operating Characteristic curve and AP is the Average Precision.

The eval.py script computes all of these metrics plus a few more under the hood such as:
- Precision Recall (PR) Curve
- Receiver Operating Characteristic (ROC) curve
- Free-Response Receiver Operating Characteristic (FROC) curve

For more information on the evaluation metrics see the [picai_eval](https://github.com/DIAGNijmegen/picai_eval) repo

An example invocation of eval.py is as follows

```bash
python eval.py --pred_path /path/to/predicted/probabilities --gt_path /path/to/groundtruth/annotations --output_path /metric_results/metrics.json
```
