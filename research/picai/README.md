#  Overview

The [PI-CAI](https://pi-cai.grand-challenge.org/) (Prostate Imaging: Cancer AI) is a collection of MRI exams to train and validate AI algorithms for detection of Clinically Significant Prostate Cancer Detection (csPCa). This folder is focused on providing examples and utilities for performing experiments on the PICAI dataset for csPCa using both centralized and a federated setup. The federated learning examples heavily leverage the [fl4health package](/README.md) to conveniently apply state-of-the-art FL techniques to real world datasets. To this end, there are the following examples:
- [U-Net on PICAI with Centralized Setup](/research/picai/central)
- [U-Net on PICAI with FedAvg](/research/picai/fedavg)
- [nnUNet scripts for PICAI](/research/picai/nnunet_scripts)
- [MONAI autosegmentation pipeline scripts for PICAI](/research/picai/monai_scripts)
- [Custom FL nnUNet for PICAI](/research/picai/fl_nnunet)

In addition to scripts to run experiments, there are a variety of utilities for [data preprocessing and augmentation](/research/picai/preprocessing), [modeling](/research/picai/model_utils.py) and [centralized baseline experiments](/research/picai/single_node_trainer.py).


## Development Requirements

For development and testing, we use [uv](https://docs.astral.sh/uv/) for dependency management. The library dependencies and those for development and testing are listed in the `pyproject.toml` file. You may use whatever virtual environment management tool that you would like. These include conda, uv itself, and virtualenv. uv is also used to produce our releases, which are managed and automated by GitHub.

The easiest way to create and activate a virtual environment is to use uv, which will automatically create a `.venv/` directory:
```bash
cd path/to/fl4health
pip install uv
uv sync --extra picai
source .venv/bin/activate
```

This will setup an environment with the proper dependencies to run the provided scripts out of the box. For more information about environment configuration, please refer to the [documentation](/CONTRIBUTING.md).

## Data
### Important Definitions
Below are the terms we will use to define model outputs at various stages of
post-processing. They are ordered sequentially from least to most processed
- **Logits:** The outputs of the model <ins>prior</ins> to the activation function. Values are unconstrained (-inf, inf)
- **Probabilities:** The outputs of the model <ins>after</ins> a normalizing activation function such as softmax or sigmoid. Values are constrained to the range (0, 1). An example for a 2d image is shown below.

<p align="center">
  <img src="/examples/nnunet_example/assets/probs.png" width="300">
</p>

- **Detection Maps:** Model predictions that contain an arbitrary number of distinct detected segmentation volumes derived from the output probabilities. Values are constrained to range [0, 1]. Example for a 2d image is shown below Detected segmentation volumes are defined as:
  - Each detected volume is a connected component that must be non-connected and non-overlapping (mutually exclusive) with other volumes of the same class. (Therefore detection maps for multiclass segmentation must be one hot encoded)
  - Each pixel/voxel within a volume must have the same predicted probability. Therefore there is a single confidence/likelihood score for each volume.
  - Detected segmentation volumes typically have a minimum size.
  - The [report guided annotation](https://github.com/DIAGNijmegen/Report-Guided-Annotation) is a common api used for deriving detection maps from model output probabilities

<p align="center">
  <img src="/examples/nnunet_example/assets/detmap.png" width="300">
</p>

- **Predicted Annotations/Segmentations**: Model predictions that have been thresholded to contain only class labels. If one hot encoded they must be binary {0, 1} or boolean {False, True} tensors. If not one hot encoded they must be be tensors containing only integers that represent the class labels (eg. constrained to {0, 1, 2, ...}). An example of a binary or one-hot-encoded predicted annotation is shown below

<p align="center">
  <img src="/examples/nnunet_example/assets/seg.png" width="300">
</p>

Below are some terms we will use to differentiate how models are combined

- **Merging:** Merging models refers to combining the weights of several different models in some way to obtain a single model that can make predictions on data with the same format as the original models. This is distinct from federated learning in the sense that no training is traditionally done in this regime.
- **Ensembling:** Ensembling refers to combining the outputs of several different models in some way to achieve a single agreed upon prediction
