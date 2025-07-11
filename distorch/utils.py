import dataclasses
import functools
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
        value = x.new_zeros(size=())
    elif next_index <= n - k:
        interp = 1 - (next_index - position)
        value = torch.amin(x) * interp
    else:
        adjusted_q = (position - (n - k)) / (k - 1)
        value = torch.quantile(x, q=adjusted_q)
    value.squeeze_()
    assert value.ndim == 0
    return value


def batchify_n_args(n: Optional[int] = None):
    def batchify_input_output(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            if n is None:
                to_batchify: tuple[Tensor, ...] = args
                args = ()
            else:
                if n <= len(args):
                    to_batchify, args = args[:n], args[n:]
                else:
                    kwargs_keys = tuple(kwargs.keys())  # kwargs keep order since Python 3.6
                    to_batchify = args + tuple(kwargs.pop(k) for k in kwargs_keys[:n - len(args)])
                    args = ()
            assert all(isinstance(t, Tensor) for t in to_batchify)

            shape = to_batchify[0].shape
            ndim = to_batchify[0].ndim
            if ndim == 1:
                raise ValueError(f'Provided tensors have 1 dim ({shape}), should be at least 2.')

            if ndim == 2:
                to_batchify = tuple(t.unsqueeze(0) for t in to_batchify)
            elif ndim > 4:
                batch_shape = shape[:-3]
                to_batchify = tuple(t.flatten(start_dim=0, end_dim=-4) for t in to_batchify)

            output = f(*to_batchify, *args, **kwargs)

            if ndim == 2 or ndim > 4:
                debatchify = (lambda t: t.squeeze(0)) if ndim == 2 else (lambda t: t.unflatten(0, batch_shape))
                if isinstance(output, Tensor):
                    output = debatchify(output)
                elif isinstance(output, (list, tuple)):
                    output = type(output)(map(debatchify, output))
                elif isinstance(output, dict):
                    output = {k: debatchify(v) for k, v in output.items()}
                elif dataclasses.is_dataclass(output):
                    for field in dataclasses.fields(output):
                        setattr(output, field.name, debatchify(getattr(output, field.name)))

            return output

        return wrapper

    return batchify_input_output
