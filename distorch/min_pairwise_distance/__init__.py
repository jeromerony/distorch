import importlib.util

import torch
from torch import Tensor

use_pykeops = importlib.util.find_spec('pykeops') is not None
use_triton = importlib.util.find_spec('triton') is not None

if use_pykeops:
    from pykeops.torch import Vi, Vj

if use_triton:
    from .cuda import min_sqdist


def minimum_distances(elem1: Tensor, elem2: Tensor) -> Tensor:
    if elem1.size(0) == 0:
        min_dists = elem1.new_zeros(size=(1,))
    elif use_pykeops and elem1.is_cuda:
        min_dists = Vi(elem1).sqdist(Vj(elem2)).min(dim=1).squeeze(1).sqrt_()
    elif use_triton and elem1.is_cuda:
        min_dists = min_sqdist(elem1, elem2).sqrt_()
    else:  # defaults to native
        min_dists = torch.cdist(elem1, elem2).amin(dim=1)

    return min_dists
