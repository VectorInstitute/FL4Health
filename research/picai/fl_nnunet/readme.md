# Federated Learning with nnUNet

## Important Defenitions
Below are the terms we will use to define model outputs at various stages of
post-processing. They are ordered sequentially from least to most processed

- **Logits:** The outputs of the model <ins>prior</ins> to the activation function. Values are unconstrained (-inf, inf)
- **Probabilities:** The outputs of the model <ins>after</ins> a normalizing activation function such as softmax or sigmoid. Values are constrained to the range (0, 1)
- **Detection Maps:** Model predictions that contain an arbitrary number of distinct detected segmentation volumes derived from the output probabilities. Values are constrained to range [0, 1]. Detected segmentation volumes are defined as:
  - Each detected volume is a connected component that must be non-connected and non-overlapping with other volumes of the same class. (Therefore detection maps for multiclass segementation must be one hot encoded)
  - Each pixel/voxel within a volume must have the same predicted probability. Therefore there is a single confidence/likelihood score for each volume.
  - Detected segmentation volumes typically have a minimum size.
  - The [report guided annotation](https://github.com/DIAGNijmegen/Report-Guided-Annotation) is a common api used for deriving detection maps from model output probabilities
- **Predicted Annotations**: Model predictions that have been thresholded to contain only class labels. If one hot encoded they must be binary {0, 1} or boolean {False, True} tensors. If not one hot encoded they must be be tensors containing only integers that represent the class labels (eg. constrained to {0, 1, 2, ...}).
