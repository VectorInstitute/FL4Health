# for testing
import torch.nn as nn
import torch
import math
# from examples.models.cnn_model import Net, MnistNet, FEMnistNet
# from fl4health.servers.secure_aggregation_utils import vectorize_model, vectorize_model_old, unvectorize_model, unvectorize_model_old
from fl4health.privacy_mechanisms.slow_discrete_gaussian_mechanism import generate_random_sign_vector, get_exponent
from fl4health.privacy_mechanisms.slow_discrete_gaussian_mechanism import (
    pad_zeros,
    randomized_rounding,
    calculate_l2_upper_bound,
    clip_vector
)

from fl4health.privacy_mechanisms.discrete_gaussian_mechanism import (
    generate_discrete_gaussian_vector
)

from fl4health.privacy_mechanisms.discrete_gaussian_mechanism import (
    fwht,
    shift_transform,
    shift_transform_torch
)

import fl4health.privacy_mechanisms.discrete_gaussian_mechanism
from fl4health.privacy.distributed_discrete_gaussian_accountant import get_heuristic_granularity

import numpy as np

# from torch.nn.utils import vector_to_parameters, parameters_to_vector

def fwht_normalized(x: torch.Tensor) -> torch.Tensor:
    """
    Applies normalized Fast Walsh-Hadamard Transform (FWHT) to 1D or 2D tensor.
    
    If input is 1D (shape: [d]), returns shape [d].
    If input is 2D (shape: [batch_size, d]), returns shape [batch_size, d].
    
    Requirements:
        - d must be a power of 2
        - Output is normalized: H_d x / sqrt(d)
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)  # make it 2D temporarily

    batch_size, d = x.shape
    assert (d & (d - 1)) == 0, "Input dimension must be a power of 2"

    h = 1
    x = x.clone()
    while h < d:
        # Shape manipulation for pairwise butterfly updates
        x = x.view(batch_size, -1, h * 2)
        a, b = x[:, :, :h], x[:, :, h:]
        x = torch.cat([a + b, a - b], dim=2)
        h *= 2

    x = x.view(batch_size, d)
    return x / d ** 0.5  # Normalize

def fwht(x: torch.Tensor):
    dim=x.size()[0]
    assert math.log2(dim).is_integer()
    log2 = int(math.ceil(math.log2(dim)))
    h_2x2 = torch.tensor([[1.0, 1.0], [1.0, -1.0]]).to(dtype=torch.float64,device=x.device)
    permutation = torch.tensor([0,2,1]).to(device=x.device)

    def _hadamard_step(x, dim):
        x_shape = x.size()
        x = x.reshape(-1, 2)
        #print(x)
        x = torch.matmul(x, h_2x2)
        x = x.view(-1, x_shape[0] // 2, 2)
        x = torch.transpose(x,2,1)
        x = x.reshape(x_shape)
        #print(x)
        return x 

    #x = x.reshape(-1, 2, dim // 2)
    index = torch.tensor(0)
    def cond(i, x):
        return i < log2

    def body(i, x):
        return i + 1, _hadamard_step(x, dim)

    while cond(index,log2):
        index,x = body(index,x)
        
    xt2 = x.view(-1, dim)
    # return xt2[0] / torch.sqrt(torch.tensor(dim, device=xt2[0].device))
    return xt2[0] / dim ** 0.5
    # xt2 = xt2.tolist()[0]
    
def client_procedure(i, x, clip, g, sign_vector, noise_multiplier, b, modulo, device):
    print(f"client {i} calculation")
    x_clipped = clip_vector(vector=x, clip=clip, granularity=g).to(device)

    print(f"vector clipped: {g* x_clipped}")

    vector = sign_vector.to(x_clipped.device) * x_clipped
    vector = fwht(vector)
    
    print(f"vector after fwht: {vector}")
    l2_upper_bound = calculate_l2_upper_bound(clip, g, 4, b)

    print(f"l2 upper bound: {l2_upper_bound}")

    vector = randomized_rounding(vector, l2_upper_bound)

    print(f"vector before rounding: {vector}")

    v = (noise_multiplier / g) ** 2
    noise_vector = torch.from_numpy(generate_discrete_gaussian_vector(dim=4, variance=v)).to(dtype=torch.float32, device=device)

    print(f"noise scale: {v}")
    vector += noise_vector

    vector = vector % modulo

    print(f"vector after rounding: {vector}")

    return vector

def server_procedure(xs, sign_vector, modulo, g, device):
    num = len(xs)
    aggregated = torch.stack(xs).sum(dim=0) % modulo

    print(f"aggregated: {aggregated}")

    vector = shift_transform_torch(aggregated, modulo)

    print(f"shifted: {vector}")

    vector = fwht(vector).to(device)

    print(f"vector after fwht: {vector}")
    vector = g * sign_vector.to(device) * vector.to(device)

    vector /= num

    print(f"vector to return: {vector}")
    return vector



if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sign_vector = generate_random_sign_vector(dim=4,).to(device)

    x = torch.tensor([-3.2, 2.1, 3.6, 40.198])
    clip = 40
    bits = 13
    modulo = 2 ** bits
    noise_multiplier = 0.007
    # g = get_heuristic_granularity(noise_multiplier, clip, bits, 3, 4)
    g = 0.1
    print(f"granularity: {g}")
    b = 0.06

    vec1 = client_procedure(1, x, clip, g, sign_vector, noise_multiplier, b, modulo, device)
    vec2 = client_procedure(2, x, clip, g, sign_vector, noise_multiplier, b, modulo, device)
    vec3 = client_procedure(3, x, clip, g, sign_vector, noise_multiplier, b, modulo, device)

    xs = [vec1, vec2, vec3]

    res = server_procedure(xs, sign_vector, modulo, g, device)

    print(f"ground truth vector: {x}")
    