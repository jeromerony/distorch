from typing import Optional

import numpy as np
import torch
from torch import Tensor

from distorch.boundary import is_surface_vertex
from distorch.utils import generate_coordinates

use_pykeops = True
try:
    from pykeops.torch import LazyTensor
except:
    import warnings

    warnings.warn('PyKeops could not be imported, this will result in high memory usage.')
    use_pykeops = False


def euclidean_distance_transform(images: Tensor,
                                 /,
                                 element_size: Optional[tuple[int | float, ...]] = None,
                                 return_indices: bool = False) -> Tensor | tuple[Tensor, Tensor]:
    """
    Similar to `scipy.ndimage.distance_transform_edt`, but computes the distance away from the True value region.

    Parameters
    ----------
    images : Tensor
        Boolean image(s)/volume(s) for which to perform the distance transform. The distance is computed away from the
        True region.
    element_size : tuple of ints or floats
        Size of a single spatial element (pixel / voxel) along each dimension. Defaults to 1 for every dimension.

    Returns
    -------
    dist : Tensor
        The calculated distance transform.

    Examples
    --------
    >>> import torch
    >>> from distorch.distance_transform import euclidean_distance_transform
    >>> img = torch.tensor([[0, 0, 0, 0, 0],
    ...                     [0, 0, 0, 1, 0],
    ...                     [0, 1, 1, 0, 0],
    ...                     [0, 1, 0, 0, 0],
    ...                     [0, 0, 0, 0, 0]], dtype=torch.bool)
    >>> euclidean_distance_transform(img)
    tensor([[2.2361, 2.0000, 1.4142, 1.0000, 1.4142],
            [1.4142, 1.0000, 1.0000, 0.0000, 1.0000],
            [1.0000, 0.0000, 0.0000, 1.0000, 1.4142],
            [1.0000, 0.0000, 1.0000, 1.4142, 2.2361],
            [1.4142, 1.0000, 1.4142, 2.2361, 2.8284]])
    """
    ndim = images.ndim
    if ndim == 2:
        images = images.unsqueeze(0)
    if ndim >= 4:
        batch_shape = images.shape[:-3]
        images = images.flatten(start_dim=0, end_dim=-4)

    b = images.size(0)
    spatial_dims = images.shape[1:]
    max_dist = float('inf')  # by convention, infinite distance for empty image
    n = int(np.prod(spatial_dims))  # number of elements (i.e. pixels/voxels)
    device = images.device
    coords = generate_coordinates(spatial_dims, device=device, element_size=element_size)

    if use_pykeops:
        coords_i = LazyTensor(coords.reshape(1, n, 1, -1))
        coords_j = LazyTensor(coords.reshape(1, 1, n, -1))
        pairwise_distances: LazyTensor = (coords_i - coords_j).norm2()
        compat = LazyTensor(torch.where(images, 0, max_dist).reshape(b, 1, n, 1))
        dist_compat = pairwise_distances + compat
        if return_indices:
            dist, indices = dist_compat.min_argmin(dim=2)
        else:
            dist = dist_compat.min(dim=2)

    else:
        pairwise_distances: Tensor = (coords.reshape(n, 1, -1) - coords.reshape(1, n, -1)).norm(p=2, dim=2)
        dist_compat = pairwise_distances.unsqueeze(0).masked_fill(~images.reshape(b, 1, n), max_dist)
        if return_indices:
            dist, indices = dist_compat.min(dim=2)
        else:
            dist = dist_compat.amin(dim=2)

    dist = dist.reshape_as(images)
    if ndim == 2:
        dist.squeeze_(0)
    elif ndim >= 4:
        dist = dist.unflatten(0, batch_shape)

    if return_indices:
        indices = indices.reshape_as(images)
        if ndim == 2:
            indices.squeeze_(0)
        elif ndim >= 4:
            indices = indices.unflatten(0, batch_shape)
        return dist, indices
    else:
        return dist


def surface_euclidean_distance_transform(images: Tensor,
                                         /,
                                         element_size: Optional[tuple[int | float, ...]] = None) -> Tensor:
    device = images.device
    ndim = images.ndim
    if ndim == 2:
        images = images.unsqueeze(0)
    if ndim >= 4:
        batch_shape = images.shape[:-3]
        images = images.flatten(start_dim=0, end_dim=-4)

    is_vertex = is_surface_vertex(images)

    coords_shape = [s + 1 for s in images.shape[1:]]
    coords_ndim = len(coords_shape)
    coords = generate_coordinates(coords_shape, device=device, element_size=element_size)
    if use_pykeops:
        coords_i = LazyTensor(coords.reshape(-1, 1, coords_ndim))

    surface_dists = []
    for is_v in is_vertex:

        if not is_v.any():
            surface_dists.append(torch.full((), float('inf'), device=device).expand(*coords_shape))
            continue

        if use_pykeops:
            surface_vertices = LazyTensor(coords[is_v].reshape(1, -1, coords_ndim))
            pairwise_distances: LazyTensor = (coords_i - surface_vertices).norm2()
            surface_dists.append(pairwise_distances.min(dim=1).reshape(*coords_shape))

        else:
            surface_vertices = coords[is_v].reshape(1, -1, coords_ndim)
            pairwise_distances: Tensor = (coords.reshape(-1, 1, coords_ndim) - surface_vertices).norm(p=2, dim=2)
            surface_dists.append(pairwise_distances.amin(dim=1).reshape(*coords_shape))

    surface_dists = torch.stack(surface_dists, dim=0)
    if ndim == 2:
        surface_dists.squeeze_(0)
    elif ndim >= 4:
        surface_dists = surface_dists.unflatten(0, batch_shape)

    return surface_dists
