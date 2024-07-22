# Federated Learning with nnUNet

## Usage
To train nnunet models using federated learning first set up a config yaml file that has the following keys. Note that either local_epochs or local_steps can be used but not both.

```yaml
n_clients: 1
nnunet_config: 2d
nnunet_plans: /path/to/nnunet/plans/file.json
fold: 0 # Which fold of the data to use for validation
n_server_rounds: 1 # number of server rounds
local_epochs: 1 # number of epochs per server round
server_address: '0.0.0.0:8080' # Default is server is the same machine
starting_checkpoint: /home/shawn/Code/nnunet_storage/nnUNet_results/Dataset012_PICAI-debug/nnUNetTrainer_1epoch__nnUNetPlans__2d/fold_0/checkpoint_best.pth # This is currently required due to a 'bug' in flwr. I have raised an issue: https://github.com/adap/flower/issues/3770
```

After creating a config file start a server using the following command. Ensure your virtual environment has been properly set up using poetry and that you have included the 'picai' group in ```poetry install```

```bash
python -m research.picai.fl_nnunet.start_server --config-path path/to/config.yaml
```

Then start a single or multiple clients in different sessions using the following command

```bash
python -m research.picai.fl_nnunet.start_client --dataset-id 012
```

The federated training will commmence once n_clients have been instantiated.

## Important Definitions
Below are the terms we will use to define model outputs at various stages of
post-processing. They are ordered sequentially from least to most processed

- **Logits:** The outputs of the model <ins>prior</ins> to the activation function. Values are unconstrained (-inf, inf)
- **Probabilities:** The outputs of the model <ins>after</ins> a normalizing activation function such as softmax or sigmoid. Values are constrained to the range (0, 1). An example for a 2d image is shown below.

  ![alt text](images/probs.png)
- **Detection Maps:** Model predictions that contain an arbitrary number of distinct detected segmentation volumes derived from the output probabilities. Values are constrained to range [0, 1]. Example for a 2d image is shown below Detected segmentation volumes are defined as:
  - Each detected volume is a connected component that must be non-connected and non-overlapping (mutually exclusive) with other volumes of the same class. (Therefore detection maps for multiclass segementation must be one hot encoded)
  - Each pixel/voxel within a volume must have the same predicted probability. Therefore there is a single confidence/likelihood score for each volume.
  - Detected segmentation volumes typically have a minimum size.
  - The [report guided annotation](https://github.com/DIAGNijmegen/Report-Guided-Annotation) is a common api used for deriving detection maps from model output probabilities

  ![alt text](images/detmap.png)
- **Predicted Annotations**: Model predictions that have been thresholded to contain only class labels. If one hot encoded they must be binary {0, 1} or boolean {False, True} tensors. If not one hot encoded they must be be tensors containing only integers that represent the class labels (eg. constrained to {0, 1, 2, ...}). An example of a binary or one-hot-encoded predicted annotation is shown below

  ![alt text](images/annotation.png)

Below are some terms we will use to differentiate how models are combined

- **Merging:** Merging models refers to combining the weights of several different models in some way to obtain a single model that can make predictions on the same data as the original models. This is distinct from federated learning in the sense that no training is traditionally done in this regime.
- **Ensembling:** Ensembling refers to combining the outputs of several different models in some way to acheive a single agreed upon prediction
