import numpy as np
import torch
from numba import njit, prange
from torch import Tensor


@njit(parallel=True)
def _minimum_sqdistance_np(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    n = x1.shape[0]
    min_sqdists = np.empty(n, dtype=x1.dtype)
    for i in prange(n):
        min_sqdists[i] = np.square(x2 - x1[i]).sum(axis=1).min()
    return min_sqdists


def min_sqdist_numba(x1: Tensor, x2: Tensor) -> Tensor:
    n, d1 = x1.shape
    m, d2 = x2.shape
    assert d1 == d2
    min_sqdistances = _minimum_sqdistance_np(np.from_dlpack(x1), np.from_dlpack(x2))
    return torch.from_numpy(min_sqdistances)
