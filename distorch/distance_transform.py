import numpy as np
import torch
from torch import Tensor

from distorch.surface import is_surface_vertex

use_pykeops = True
try:
    from pykeops.torch import LazyTensor
except:
    import warnings

    warnings.warn('PyKeops could not be imported, this will result in high memory usage.')
    use_pykeops = False


def euclidean_distance_transform(images: Tensor) -> Tensor:
    """
    Similar to `scipy.ndimage.distance_transform_edt`, but computes the distance away from the True value region.
    TODO: add comprehensive docstring
    """
    b = images.size(0)
    spatial_dims = images.shape[1:]
    max_dist = float('inf')  # by convention, infinite distance for empty image
    n = int(np.prod(spatial_dims))  # number of elements (i.e. pixels/voxels)
    device = images.device

    aranges = (torch.arange(s, device=device, dtype=torch.float) for s in spatial_dims)
    pos = torch.stack(torch.meshgrid(*aranges, indexing='ij'), dim=-1)

    if use_pykeops:
        pos_i = LazyTensor(pos.reshape(1, n, 1, -1))
        pos_j = LazyTensor(pos.reshape(1, 1, n, -1))
        pairwise_distances: LazyTensor = (pos_i - pos_j).norm2()
        compat = LazyTensor(torch.where(images, 0, max_dist).reshape(b, 1, n, 1))
        dist = (pairwise_distances + compat).min(dim=2)

    else:
        pairwise_distances: Tensor = (pos.reshape(n, 1, -1) - pos.reshape(1, n, -1)).norm(p=2, dim=2)
        dist = pairwise_distances.unsqueeze(0).masked_fill(~images.reshape(b, 1, n), max_dist).amin(dim=2)

    dist = dist.reshape_as(images)
    return dist


def surface_euclidean_distance_transform(images: Tensor) -> Tensor:
    device = images.device
    is_vertex = is_surface_vertex(images)

    coords_shape = [s + 1 for s in images.shape[1:]]
    coords_ndim = len(coords_shape)
    aranges = [torch.arange(s, device=device, dtype=torch.float) for s in coords_shape]
    coords = torch.stack(torch.meshgrid(*aranges, indexing='ij'), dim=-1)
    if use_pykeops:
        coords_i = LazyTensor(coords.reshape(-1, 1, coords_ndim))

    surface_dists = []
    for is_v in is_vertex:

        if use_pykeops:
            surface_vertices = LazyTensor(coords[is_v].reshape(1, -1, coords_ndim))
            pairwise_distances: LazyTensor = (coords_i - surface_vertices).norm2()
            surface_dists.append(pairwise_distances.min(dim=1).reshape(*coords_shape))

        else:
            surface_vertices = coords[is_v].reshape(1, -1, coords_ndim)
            pairwise_distances: Tensor = (coords.reshape(-1, 1, coords_ndim) - surface_vertices).norm(p=2, dim=2)
            surface_dists.append(pairwise_distances.amin(dim=1).reshape(*coords_shape))

    surface_dists = torch.stack(surface_dists, dim=0)
    return surface_dists
