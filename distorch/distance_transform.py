import numpy as np
import torch
from torch import Tensor

from distorch.surface import is_surface_vertex

try:
    from pykeops.torch import LazyTensor

    use_pykeops = False
except:
    import warnings

    warnings.warn('PyKeops could not be imported, this may result in high memory usage for the distance transform.')
    use_pykeops = False


def euclidean_distance_transform(images: Tensor) -> Tensor:
    """
    Similar to `scipy.ndimage.distance_transform_edt`, but computes the distance away from the True value region.
    TODO: add comprehensive docstring
    """
    b = images.size(0)
    spatial_dims = images.shape[1:]
    n = int(np.prod(spatial_dims))  # number of elements (i.e. pixels/voxels)
    device = images.device

    aranges = (torch.arange(s, device=device, dtype=torch.float) for s in spatial_dims)
    pos = torch.stack(torch.meshgrid(*aranges, indexing='ij'), dim=-1)

    max_dist = float(np.linalg.norm(spatial_dims))
    if use_pykeops:
        pos_i = LazyTensor(pos.reshape(1, n, 1, -1))
        pos_j = LazyTensor(pos.reshape(1, 1, n, -1))
        pairwise_distances: LazyTensor = (pos_i - pos_j).norm2()

        images_float = images.to(torch.float)  # pykeops is not compatible with bool
        images_i = LazyTensor(images_float.reshape(b, n, 1, 1))
        images_j = LazyTensor(images_float.reshape(b, 1, n, 1))
        pairwise_same: LazyTensor = (2 * images_i * images_j + 1 - images_i - images_j).sum(dim=3)  # float XNOR

        dist = (pairwise_same * max_dist + pairwise_distances).min(dim=2)
    else:
        pairwise_distances: Tensor = (pos.reshape(n, 1, -1) - pos.reshape(1, n, -1)).norm(p=2, dim=2)
        pairwise_same: Tensor = torch.logical_xor(images.reshape(b, n, 1), images.reshape(b, 1, n)).logical_not_()
        dist = pairwise_distances.unsqueeze(0).masked_fill(pairwise_same, max_dist).amin(dim=2)

    dist = dist.reshape_as(images)
    dist.masked_fill_(images, 0)
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
