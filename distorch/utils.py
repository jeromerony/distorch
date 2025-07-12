import dataclasses
import functools
import inspect
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


def batchify_args(*args_to_batchify: str):
    def batchify_input_output(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            # transform all args to kwargs
            if args:
                sig = inspect.signature(f)
                param_names = list(sig.parameters.keys())

                for i, arg in enumerate(args):
                    if i < len(param_names):
                        kwargs[param_names[i]] = arg
                    else:
                        raise TypeError(f"Too many positional arguments for {f.__name__}")

            shape = kwargs[args_to_batchify[0]].shape
            assert all(kwargs[k].shape == shape for k in args_to_batchify)
            ndim = len(shape)
            if ndim < 2:
                raise ValueError(f'Provided tensors have fewer than 2 dims ({shape}), should be at least 2.')

            batchify_func = debatchify_func = lambda t: t
            if ndim == 2:
                batchify_func = lambda t: t.unsqueeze(0)
                debatchify_func = lambda t: t.squeeze(0)
            elif ndim > 4:
                batch_shape = shape[:-3]
                batchify_func = lambda t: t.flatten(start_dim=0, end_dim=-4)
                debatchify_func = lambda t: t.unflatten(0, batch_shape)

            for name in args_to_batchify:
                kwargs[name] = batchify_func(kwargs[name])

            output = f(**kwargs)

            if ndim == 2 or ndim > 4:
                if isinstance(output, Tensor):
                    output = debatchify_func(output)
                elif isinstance(output, (list, tuple)):
                    output = type(output)(map(debatchify_func, output))
                elif isinstance(output, dict):
                    output = {k: debatchify_func(v) for k, v in output.items()}
                elif dataclasses.is_dataclass(output):
                    for field in dataclasses.fields(output):
                        setattr(output, field.name, debatchify_func(getattr(output, field.name)))

            return output

        return wrapper

    return batchify_input_output
