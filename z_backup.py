import numpy as np
import time, math, timeit
from numba import jit, prange
@jit(nopython=True)
def fwht(a) -> None:
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

a = np.random.rand(2**27)
for i in range(10):
    t0 = time.perf_counter()
    fwht(a)
    t1 = time.perf_counter()
    print(i, t1-t0)

