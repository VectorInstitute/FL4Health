from typing import Any, Tuple

import numpy as np
import torch
from numpy import linalg

np.random.seed(0)
torch.manual_seed(0)

torch_matrix = torch.rand(4, 4)
np_matrix = np.random.rand(4, 4)

pos_def_torch_mat = torch.mul(torch_matrix, torch_matrix.T)
pos_def_np_mat = np.matmul(np_matrix, np_matrix.T)


def compute_mult_and_computeeigen_info(numpy_mat: np.ndarray, torch_matrix: torch.Tensor) -> Tuple[Any, ...]:
    mat = np.matmul(numpy_mat, torch_matrix)
    return linalg.eig(mat)


eigen_vals, eigen_vectors = compute_mult_and_computeeigen_info(np_matrix, torch_matrix)

print(f"Eigen values: {eigen_vals}")
print(f"Eigen vectors: {eigen_vectors}")
