import torch


def infer_label_dim(tensor1: torch.Tensor, tensor2: torch.Tensor) -> int:
    """
    Infers the label dimension given two related tensors of different shapes.

    Generally useful for inferring the label dimension when one tensor is vector-encoded and the other is not. The
    label dimension is inferred by looking for dimensions that either are not the same size, or are not present in
    tensor 2.

    Args:
        tensor1 (torch.Tensor): The reference tensor. Must have the same number of dimensions as tensor 2, or have
            exactly 1 more dimension (the label dimension).
        tensor2 (torch.Tensor): The non-reference tensor.

    Raises:
        AssertionError: If the the label dimension cannot be inferred without ambiguity. For example if a dimension
            next to the label dimension has the same size.

    Returns:
        (int): Index of the dimension along tensor 1 that corresponds to the label dimension.
    """
    assert tensor1.shape != tensor2.shape, (
        f"Could not infer the label dimension of tensors with the same shape: {tensor1.shape}"
    )

    assert 0 <= (tensor1.ndim - tensor2.ndim) <= 1, (
        f"Could not infer the label dimension of tensors with shapes: tensor1: {tensor1.shape}), tensor 2: "
        f"({tensor2.shape}). Expected tensor1 to be larger than tensor2 by at most 1 dimension."
    )

    # Infer label dimension.
    idx2 = 0
    candidate_label_dims = []
    # Iterate through dimensions of tensor1 and compare size to corresponding dimension of tensor2
    # If tensor1 and tensor2 have the same number of dimensions, we are looking for axes in the same position with
    # different sizes. If tensor1 has an extra dimension, then we are looking for an axes with likely unique size not
    # present in tensor2
    for idx1 in range(tensor1.ndim):
        if idx2 >= tensor2.ndim:
            # If tensor1 has an extra dimension and all previous dimensions were the same shape, then the last
            # dimension must be the additional dimension
            candidate_label_dims.append(idx1)
        elif tensor1.shape[idx1] == tensor2.shape[idx2]:
            # If the dimensions have the size then they are likely not the label dimension
            idx2 += 1
        else:
            candidate_label_dims.append(idx1)
            if tensor1.ndim == tensor2.ndim:
                # If tensor1 and tensor2 have the same number of dimensions, then we are looking for axes in the same
                # position with different sizes. Hence we proceed to the next index of tensor 2. Otherwise, the
                # non-matching dimension in tensor1 is an extra dimension, and so we want to act as if we skipped it
                # and ensure the rest of the shape is the same.
                idx2 += 1

    assert len(candidate_label_dims) == 1, (
        f"Could not infer the label dimension of tensors with shapes: ({tensor1.shape}), ({tensor2.shape}). "
        "Found multiple axes that could be the label dimension."
    )
    label_dim = candidate_label_dims[0]

    # Cover edge case where dim adjacent to label dim has the same size. We will mistakenly resolve only a single
    # candidate label dimension when technically it is ambiguous. An example is
    #   tensor_1.shape = (5, 5, 3) and tensor_2.shape = (5, 3).
    # The label dimension could be the first or the second.
    if tensor1.ndim > tensor2.ndim and label_dim > 0:
        assert tensor1.shape[label_dim] != tensor1.shape[label_dim - 1], (
            f"Could not infer the label dimension of tensors with shapes: ({tensor1.shape}), ({tensor2.shape}). "
            "A dimension adjacent to the label dimension appears to have the same size."
        )

    # If tensors have same ndim but different shapes, then this only works if label dim was empty for one of them
    if tensor1.ndim == tensor2.ndim:
        assert (tensor1.shape[label_dim] == 1) or (tensor2.shape[label_dim]) == 1, (
            f"Could not infer the label dimension of tensors with shapes: ({tensor1.shape}), ({tensor2.shape}). "
            "The inferred candidate dimension has different sizes on each tensor, was expecting one to be empty."
        )
    return label_dim


def map_label_index_tensor_to_one_hot(
    label_index_tensor: torch.Tensor, target_shape: torch.Size, label_dim: int
) -> torch.Tensor:
    """
    Maps the provided ``label_index_tensor``, which has label indices at the provided ``label_dim``. In the tensor,
    this dimension should be "empty," i.e. have size 1. This function uses the shape provided by ``target_shape`` to
    expand the label indices into one-hot encoded vectors in that dimension according the the size of the target
    dimension at ``label_dim`` in ``target_shape``. For example, if ``label_index_tensor`` has shape
    ``(64, 10, 10, 1)``, ``label_dim = 3``, and ``target_shape = (64, 10, 10, 4)``, then the new shape should be
    ``(64, 10, 10, 4)`` with ``[i, j, k, :]`` being a one-hot vector of length ``4``.

    Args:
        label_index_tensor (torch.Tensor): Tensor to have label_dim dimension one-hot encoded accounting to
            ``target_shape`` and the indices of ``label_index_tensor`` in the ``label_dim``
        target_shape (torch.Size): Shape we want to transform ``label_index_tensor`` to. Mainly used to establish the
            length of the one-hot encodings
        label_dim (int): Dimension to one-hot encode.

    Returns:
        (torch.Tensor): Tensor with one-hot encoded ``label_dim``.
    """
    label_index_tensor_shape = label_index_tensor.shape

    assert label_dim < len(label_index_tensor_shape), (
        f"Label dim: {label_dim} too large for target shape: {label_index_tensor_shape}"
    )

    label_dim_of_tensor = label_index_tensor.shape[label_dim]
    assert label_dim_of_tensor == 1, (
        f"Expected label_dim {label_dim} of label_index_tensor to be of size 1, but got {label_dim_of_tensor}"
    )

    one_hot_encoded_tensor = torch.zeros(target_shape, device=label_index_tensor.device)
    one_hot_encoded_tensor.scatter_(label_dim, label_index_tensor.to(torch.int64), 1)
    return one_hot_encoded_tensor


def align_pred_and_target_shapes(
    preds: torch.Tensor, targets: torch.Tensor, label_dim: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    If necessary, attempts to correct shape mismatches between the given tensors by inferring which one to
    one-hot-encode.

    **NOTE**: If both preds and targets have the same shape then nothing needs to be modified. If there is a mismatch,
    it is assumed that the labels in one of the predictions or the targets are vector encoded in some way
    (either one-hot or soft) while the other is label index encoded.

    If one is vector encoded but not the other, then both are returned as vector encoded tensors.

    **NOTE**: This function **ASSUMES** label-index encoding if the shapes are misaligned. This assumption doesn't
    necessarily hold in binary classification settings where continuous values might be used to indicate the positive
    label by default. As such, this function should not be used for those types of tensors.

    For example, consider a problem with 3 label classes with the preds vector encoded and the targets label encoded

    ```python
    preds = torch.Tensor([[0.1, 0.2, 0.7], [0.9, 0.1, 0.0]])

    targets = torch.Tensor([[2], [1]])
    ```

    preds has shape ``(2, 3)`` and targets has shape ``(2, 1)``. This function will convert targets to a
    one-hot-encode tensor with contents

    ```python
    targets = torch.Tensor([0, 0, 1], [0, 1, 0])
    ```

    Args:
        preds (torch.Tensor): The tensor with model predictions.
        targets (torch.Tensor): The tensor with model targets.
        label_dim (int | None): Index of the label dimension. If left as None then this method attempts to infer
            the label dimension if it is needed.

    Returns:
        (tuple[torch.Tensor, torch.Tensor]): The pred and target tensors respectively now ensured to have the same
            shape.
    """
    # Shapes are already aligned.
    if preds.shape == targets.shape:
        return preds, targets

    # Run this assertion before in case label dim is defined.
    assert abs(preds.ndim - targets.ndim) <= 1, (
        f"Can not align pred and target tensors with shapes {preds.shape}, {targets.shape}"
    )

    # If shapes are different then we assume one tensor has vector encoded labels and the other is label index encoded
    # and will be mapped to one-hot-encoded format.
    if preds.ndim > targets.ndim:
        # Preds must be vector encoded and targets are not
        # Determine label dimension. Preds is the first/reference tensor because its shape is larger, if not provided
        label_dim = infer_label_dim(preds, targets) if label_dim is None else label_dim
        targets = targets.unsqueeze(label_dim)
        one_hot_targets = map_label_index_tensor_to_one_hot(targets, preds.shape, label_dim)
        return preds, one_hot_targets

    if preds.ndim < targets.ndim:
        # Targets must be vector encoded and preds are not
        # Determine label dimension. Targets is the first/reference tensor because its shape is larger, if not provided
        label_dim = infer_label_dim(targets, preds) if label_dim is None else label_dim
        preds = preds.unsqueeze(label_dim)
        one_hot_preds = map_label_index_tensor_to_one_hot(preds, targets.shape, label_dim)
        return one_hot_preds, targets

    # If we're here, the shapes are the same size, but differ in at least one dimension
    # Determine label dimension, if not provided. Because their shapes are the same length, order doesn't matter
    label_dim = infer_label_dim(preds, targets) if label_dim is None else label_dim
    if preds.shape[label_dim] < targets.shape[label_dim]:
        # We need to one-hot the preds tensor
        return map_label_index_tensor_to_one_hot(preds, targets.shape, label_dim), targets
    return preds, map_label_index_tensor_to_one_hot(targets, preds.shape, label_dim)
