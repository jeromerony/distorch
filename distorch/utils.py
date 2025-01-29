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
