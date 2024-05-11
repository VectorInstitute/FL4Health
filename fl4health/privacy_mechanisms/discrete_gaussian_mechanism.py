import numpy as np
from numba import jit, prange
import time, math 
from typing import List
import torch

"""References
    [DGDP] The Discrete Gaussian for Differential Privacy
    https://proceedings.neurips.cc/paper/2020/file/b53b3a3d6ab90ce0268229151c9bde11-Paper.pdf
"""

@jit(nopython=True)
def bernoulli_exp(negative_gamma: float) -> int:
    """Draws Bernoulli sample with probability exp(negative_gamma).
    Used in discrete Gaussian sampler. See [DGDP].
    """
    g = -negative_gamma
    assert g >= 0

    if g <= 1:
        K = 1
        while np.random.binomial(n=1, p=g/K):
            K += 1
        return K % 2
    else:
        for _ in prange(int(np.floor(g))):
            if bernoulli_exp(-1) == 0:
                return 0
        return bernoulli_exp(np.floor(g)-g)
    
@jit(nopython=True)
def discrete_gaussian_sampler(variance: float) -> int:
    """Draw a sample from discrete Gaussian random variable with given variance.
    See [DGDP].
    """
    assert variance > 0
    t = 1 + int(np.floor(np.sqrt(variance)))
    while True:
        U = np.random.randint(low=0, high=t)
        if bernoulli_exp(-U/t) == 0:
            continue
        V = 0
        while bernoulli_exp(-1):
            V += 1
        B = np.random.randint(low=0, high=2)
        if B and not U and not V:
            continue
        Z = (1-2*B) * (U+t*V)
        gamma = (np.absolute(Z)-variance/t)**2 / (2*variance)
        if not bernoulli_exp(-gamma):
            continue 
        return Z
    
@jit(nopython=True, parallel=True)
def generate_discrete_gaussian_vector(dim: int, variance: float) -> np.array:
    """Assume dim is a power of 2."""
    return np.array([discrete_gaussian_sampler(variance) for _ in prange(dim)])

@jit(nopython=True)
def wiki_fwht(a) -> None:
    """In-place Fast Walsh–Hadamard Transform of array a."""
    h = 1
    while h < len(a):
        # perform FWHT
        for i in prange(0, len(a), h * 2):
            for j in prange(i, i + h):
                x = a[j]
                y = a[j + h]
                a[j] = x + y
                a[j + h] = x - y
        # normalize and increment
        a /= math.sqrt(2)
        h *= 2
    return a

def fwht(x):
    dim=x.size()[0]
    assert math.log2(dim).is_integer()
    log2 = int(math.ceil(math.log2(dim)))
    h_2x2 = torch.tensor([[1.0, 1.0], [1.0, -1.0]]).to(dtype=torch.float64)
    permutation = torch.tensor([0,2,1])

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
    return xt2[0] / torch.sqrt(torch.tensor(dim))
    # xt2 = xt2.tolist()[0]
    
@jit(nopython=True)
def shift_transform(vect: np.array, r: int) -> np.array:
    m = r % 2
    half = r // 2

    assert m  == 0

    for i in prange(vect.size):
        component = vect[i] % r
        if 0 <= component <= half:
            vect[i] = component
        else:
            vect[i] = component - r

    return vect

@jit(nopython=True)
def modular_clipping(vector: np.array, a: int, b: int) -> np.array:
    return vector - (b-a) * np.floor((vector-a)/(b-a))
