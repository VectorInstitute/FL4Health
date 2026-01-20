import re

import pytest
import torch

from fl4health.metrics.utils import align_pred_and_target_shapes, infer_label_dim, map_label_index_tensor_to_one_hot


MULTIPLE_AXES_PATTERN = re.compile("Found multiple axes that could be the label dimension", flags=re.IGNORECASE)
SAME_DIM_DIFFERENT_SIZE_PATTERN = re.compile(
    "The inferred candidate dimension has different sizes on each tensor, was expecting one to be empty.",
    flags=re.IGNORECASE,
)
AXES_OF_SAME_SIZE_PATTERN = re.compile(
    "A dimension adjacent to the label dimension appears to have the same size.", flags=re.IGNORECASE
)
SHAPES_LENGTHS_ARE_PROBLEMATIC = re.compile(
    "Expected tensor1 to be larger than tensor2 by at most 1 dimension.", flags=re.IGNORECASE
)
AMBIGUOUS_LABEL_DIMENSIONS = re.compile("Found multiple axes that could be the label dimension.", flags=re.IGNORECASE)
SAME_SHAPE_PATTERN = re.compile(
    "Could not infer the label dimension of tensors with the same shape", flags=re.IGNORECASE
)


def test_infer_label_dim() -> None:
    preds = torch.randn((2, 5, 3))
    targets = torch.randn((2, 3))
    assert infer_label_dim(preds, targets) == 1

    preds = torch.randn((2, 5, 3))
    targets = torch.randn((2, 1, 3))
    assert infer_label_dim(preds, targets) == 1

    preds = torch.randn((2, 3, 5))
    targets = torch.randn((2, 3, 1))
    assert infer_label_dim(preds, targets) == 2

    preds = torch.randn((2, 3, 5))
    targets = torch.randn((2, 3))
    assert infer_label_dim(preds, targets) == 2


def test_infer_label_dim_failures() -> None:
    # Tests that the proper exceptions are raised when attempting to infer the label dimension for
    # pred and target tensors.

    # Tensors of same shape are not eligible for inferring the label dimension
    preds = torch.randn((2, 3, 4))
    targets = torch.randn((2, 3, 4))
    with pytest.raises(Exception, match=SAME_SHAPE_PATTERN):
        infer_label_dim(preds, targets)

    # Tensors with shape length disparity of more than 1 are ineligible
    preds = torch.randn((5, 6))
    targets = torch.randn((5, 6, 7, 8))
    with pytest.raises(Exception, match=SHAPES_LENGTHS_ARE_PROBLEMATIC):
        infer_label_dim(preds, targets)

    preds = torch.randn((2, 5))
    targets = torch.randn((2, 5, 3, 4))
    with pytest.raises(Exception, match=SHAPES_LENGTHS_ARE_PROBLEMATIC):
        infer_label_dim(preds, targets)

    preds = torch.randn((2, 5, 3, 4))
    targets = torch.randn((2, 3, 4, 4))
    with pytest.raises(Exception, match=AMBIGUOUS_LABEL_DIMENSIONS):
        infer_label_dim(preds, targets)

    preds = torch.randn((5, 5, 3))
    targets = torch.randn((5, 2))
    with pytest.raises(Exception, match=AMBIGUOUS_LABEL_DIMENSIONS):
        infer_label_dim(preds, targets)

    preds = torch.randn((5, 5, 3))
    targets = torch.randn((5, 3))
    with pytest.raises(Exception, match=AXES_OF_SAME_SIZE_PATTERN):
        infer_label_dim(preds, targets)

    preds = torch.randn((5, 5, 3))
    targets = torch.randn((5, 10, 3))
    with pytest.raises(Exception, match=SAME_DIM_DIFFERENT_SIZE_PATTERN):
        infer_label_dim(preds, targets)


def get_dummy_classification_tensors(
    full_tensor_shape: tuple[int, ...], label_dim: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns 3 versions of the same dummy tensor to use for metric tests. The shape provided in the arguments
    specifies the shape of tensors with vector encoded targets. The function will create "soft" encodings
    (i.e not one-hot), "hard" encodings (i.e. one-hot), and label index encodings.

    Args:
        full_tensor_shape (tuple[int, ...]): Size of the desired tensor, assuming label dimension is vector encoded
        label_dim (int): which dimension of the tensor should be taken as the label dimension.

    Returns:
        (tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]): Tensors corresponding to different
            types of encodings. In order, soft vector encodings, one hot encodings, label index encodings.
    """
    n_classes = full_tensor_shape[label_dim]
    assert n_classes > 1, "Must have at least 2 classes"

    # Create soft vector-encoded tensor
    soft_vector_encoded = torch.rand(size=full_tensor_shape)
    soft_vector_encoded = torch.softmax(soft_vector_encoded, dim=label_dim)

    # Create label-index encoded tensor
    label_index_encoded = torch.argmax(soft_vector_encoded, dim=label_dim)

    # Create one-hot-encoded tensor, essentially threshold soft_vector_encoded tensor
    one_hot_tensor = torch.zeros(size=full_tensor_shape)
    ohe_view = label_index_encoded.view(
        (*label_index_encoded.shape[:label_dim], 1, *label_index_encoded.shape[label_dim:])
    )
    one_hot_tensor.scatter_(label_dim, ohe_view, 1)

    return one_hot_tensor, soft_vector_encoded, label_index_encoded


def test_map_label_index_tensor_to_one_hot() -> None:
    # Small example
    label_dim = 2
    # Each of the entries corresponds to a label index
    label_index_tensor = torch.Tensor([[[1, 2], [3, 0]], [[1, 1], [2, 3]], [[0, 1], [1, 1]]])  # shape: (3, 2, 2)
    label_index_tensor = label_index_tensor.unsqueeze(label_dim)
    one_hot_tensor = map_label_index_tensor_to_one_hot(label_index_tensor, torch.Size([3, 2, 4, 2]), label_dim)
    # We should have expanded the label indices into the 3rd dimension
    assert torch.allclose(one_hot_tensor[0, 0, :, 0], torch.Tensor([0, 1, 0, 0]))
    assert torch.allclose(one_hot_tensor[0, 1, :, 0], torch.Tensor([0, 0, 0, 1]))
    assert torch.allclose(one_hot_tensor[1, 1, :, 1], torch.Tensor([0, 0, 0, 1]))
    assert torch.allclose(one_hot_tensor[2, 0, :, 0], torch.Tensor([1, 0, 0, 0]))
    assert torch.allclose(one_hot_tensor[2, 1, :, 1], torch.Tensor([0, 1, 0, 0]))
    assert torch.allclose(one_hot_tensor[2, 1, :, 0], torch.Tensor([0, 1, 0, 0]))

    # Big example
    tensor_shape = (2, 3, 5, 9, 3)
    label_dim = 1
    soft_vector_encoded = torch.rand(size=tensor_shape)
    soft_vector_encoded = torch.softmax(soft_vector_encoded, dim=label_dim)

    label_index_encoded = torch.argmax(soft_vector_encoded, dim=label_dim)  # shape (2, 5, 9, 3)

    # Force soft_vector_encoded to be one-hot encoded
    one_hot_tensor = torch.zeros(size=tensor_shape)
    ohe_view = label_index_encoded.view(
        (*label_index_encoded.shape[:label_dim], 1, *label_index_encoded.shape[label_dim:])
    )
    one_hot_tensor.scatter_(label_dim, ohe_view, 1)

    label_index_encoded = label_index_encoded.unsqueeze(label_dim)
    one_hot_tensor_comparison = map_label_index_tensor_to_one_hot(
        label_index_encoded, torch.Size(tensor_shape), label_dim
    )
    # Make sure the manual one-hot encoding matches the mapped one-hot encodings.
    assert torch.allclose(one_hot_tensor, one_hot_tensor_comparison)


def test_multiclass_align() -> None:
    # Tests the auto shape alignment used by the ClassificationMetric base class for multi-class classification.

    # Create dummy preds and targets
    one_hot_tensor_preds, soft_vector_encoded_preds, label_index_encoded_preds = get_dummy_classification_tensors(
        (2, 3, 5, 9, 3), 1
    )

    (
        one_hot_tensor_targets,
        soft_vector_encoded_targets,
        label_index_encoded_targets,
    ) = get_dummy_classification_tensors((2, 3, 5, 9, 3), 1)

    # Targets should be modified to be one hot tensors and thus match that dummy, preds should be unmodified
    preds, targets = align_pred_and_target_shapes(soft_vector_encoded_preds, label_index_encoded_targets)
    assert preds.shape == targets.shape == (2, 3, 5, 9, 3)
    assert torch.isclose(preds, soft_vector_encoded_preds).all()
    assert torch.isclose(targets, one_hot_tensor_targets).all()

    # Targets should be modified to be one hot tensors and thus match that dummy, preds should be unmodified
    preds, targets = align_pred_and_target_shapes(one_hot_tensor_preds, label_index_encoded_targets)
    assert preds.shape == targets.shape == (2, 3, 5, 9, 3)
    assert torch.isclose(preds, one_hot_tensor_preds).all()
    assert torch.isclose(targets, one_hot_tensor_targets).all()

    # Preds should be modified to be one hot encoded and thus match that dummy, targets should be unmodified
    preds, targets = align_pred_and_target_shapes(label_index_encoded_preds, one_hot_tensor_targets)
    assert preds.shape == targets.shape == (2, 3, 5, 9, 3)
    assert torch.isclose(preds, one_hot_tensor_preds).all()
    assert torch.isclose(targets, one_hot_tensor_targets).all()

    # Preds should be modified to be one hot encoded and thus match that dummy, targets should be unmodified
    preds, targets = align_pred_and_target_shapes(label_index_encoded_preds, soft_vector_encoded_targets)
    assert preds.shape == targets.shape == (2, 3, 5, 9, 3)
    assert torch.isclose(preds, one_hot_tensor_preds).all()
    assert torch.isclose(targets, soft_vector_encoded_targets).all()

    # Test that if shapes are the same outputs are unchanged
    preds, targets = align_pred_and_target_shapes(one_hot_tensor_preds, soft_vector_encoded_targets)
    assert preds.shape == targets.shape == (2, 3, 5, 9, 3)
    assert torch.isclose(preds, one_hot_tensor_preds).all()
    assert torch.isclose(targets, soft_vector_encoded_targets).all()

    # Test that if shapes are the same outputs are unchanged
    preds, targets = align_pred_and_target_shapes(soft_vector_encoded_preds, one_hot_tensor_targets)
    assert preds.shape == targets.shape == (2, 3, 5, 9, 3)
    assert torch.isclose(preds, soft_vector_encoded_preds).all()
    assert torch.isclose(targets, one_hot_tensor_targets).all()


def test_align_exceptions() -> None:
    # Label dim can not be resolved if shapes differ in more than 1 dimension
    one_hot_tensor_preds, soft_vector_encoded_preds, _ = get_dummy_classification_tensors((2, 3, 5, 9, 3), 1)
    one_hot_tensor_targets, _, label_index_encoded_targets = get_dummy_classification_tensors((2, 3, 5, 9, 6), 1)

    with pytest.raises(Exception, match=MULTIPLE_AXES_PATTERN):
        # soft_vector_encoded_preds: (2, 3, 5, 9, 3), label_index_encoded_targets: (2, 5, 9, 6)
        align_pred_and_target_shapes(soft_vector_encoded_preds, label_index_encoded_targets)

    with pytest.raises(Exception, match=SAME_DIM_DIFFERENT_SIZE_PATTERN):
        # one_hot_tensor_preds: (2, 3, 5, 9, 3), one_hot_tensor_targets: (2, 3, 5, 9, 6)
        align_pred_and_target_shapes(one_hot_tensor_preds, one_hot_tensor_targets)

    # Channel dim can not be resolved if the dimension directly afterwards has the same size
    one_hot_tensor_preds, soft_vector_encoded_preds, _ = get_dummy_classification_tensors((2, 3, 3, 9, 6), 1)
    one_hot_tensor_targets, _, label_index_encoded_targets = get_dummy_classification_tensors((2, 3, 3, 9, 6), 1)

    with pytest.raises(Exception, match=AXES_OF_SAME_SIZE_PATTERN):
        # soft_vector_encoded_preds: (2, 3, 3, 9, 6), label_index_encoded_targets: (2, 3, 9, 6)
        align_pred_and_target_shapes(soft_vector_encoded_preds, label_index_encoded_targets)
