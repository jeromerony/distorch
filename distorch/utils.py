import math
from typing import Optional

import torch
from torch import Tensor
from torch._prims_common import extract_shape_from_varargs


def generate_coordinates(*size,
                         dtype: Optional[torch.dtype] = torch.float,
                         device: Optional[torch.device] = None,
                         element_size: Optional[tuple[int | float, ...]] = None) -> Tensor:
    size = extract_shape_from_varargs(size)  # reproduce behavior of torch.zeros, torch.ones, etc.
    aranges = [torch.arange(s, device=device, dtype=dtype) for s in size]
    if element_size is not None:
        assert len(element_size) == len(aranges)
        torch._foreach_mul_(aranges, element_size)
    coordinates = torch.stack(torch.meshgrid(*aranges, indexing='ij'), dim=-1)
    return coordinates


def zero_padded_nonnegative_quantile(x: Tensor, q: float, n: int) -> Tensor:
    """
    Compute the q-th quantile for a nonnegative 1d Tensor, adjusted for 0 values.
    This function is equivalent to padding `x` with 0 values such that it has a size `n`.

    Parameters
    ----------
    x : Tensor
        The 1d input tensor.
    q : float
        A scalar in the range [0, 1].
    n : int
        The size of `x` including 0 values, should verify `n >= x.size(0)`.

    Examples
    --------
    >>> x = torch.randn(3).abs_()
    >>> x
    tensor([0.3430, 1.0778, 0.5040])
    >>> zero_padded_nonnegative_quantile(x, q=0.75, n=10)
    tensor(0.2573)
    >>> import torch.nn.functional as F
    >>> x_padded = F.pad(x, (0, 7), value=0)
    >>> x_padded
    tensor([0.3430, 1.0778, 0.5040, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000])
    >>> torch.quantile(x_padded, q=0.75)
    tensor(0.2573)

    """
    k = x.size(0)
    assert n >= 1
    assert n >= k
    position = (n - 1) * q
    next_index = math.ceil(position)
    if k < 1 or next_index <= (n - k - 1):
        return x.new_zeros(size=(1,))
    elif next_index <= n - k:
        interp = 1 - (next_index - position)
        return torch.amin(x) * interp
    else:
        adjusted_q = (position - (n - k)) / (k - 1)
        return torch.quantile(x, q=adjusted_q)
