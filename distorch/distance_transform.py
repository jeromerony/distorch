from typing import Optional

import cupy as cp
import torch
from cucim.core.operations.morphology import distance_transform_edt
from torch import Tensor

from distorch.boundary import is_surface_vertex


def cucim_edt(images: Tensor, /, element_size=None) -> Tensor:
    cp_images = cp.from_dlpack(~images)
    edts = []
    for cp_img in cp_images:
        if cp_img.all():
            edts.append(cp.full_like(cp_img, float('inf'), dtype=cp.float32))
        else:
            edts.append(distance_transform_edt(cp_img, return_distances=True, sampling=element_size))
    edts = cp.stack(edts, axis=0)
    return torch.from_dlpack(edts).reshape_as(images)


def euclidean_distance_transform(images: Tensor,
                                 /,
                                 element_size: Optional[tuple[int | float, ...]] = None) -> Tensor:
    """
    Similar to `scipy.ndimage.distance_transform_edt`, but computes the distance away from the True value region.

    Parameters
    ----------
    images : Tensor
        Boolean image(s)/volume(s) for which to perform the distance transform. The distance is computed away from the
        True region.
    element_size : tuple of ints or floats
        Size of a single spatial element (pixel / voxel) along each dimension. Defaults to 1 for every dimension.
    return_indices : bool
        Optionally returns the indices of the closest True element, i.e. the one to which the distance is calculated.
        The indices are flat for memory efficiency, meaning they index over all the spatial dimensions in C order.
        To obtain unraveled indices, use `torch.unravel_index` with the appropriate shape.

    Returns
    -------
    dist : Tensor
        The calculated distance transform.
    indices : Tensor
        Flat indices of the closest True element, also known as the feature transform.

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

    dist = cucim_edt(images, element_size=element_size)

    if ndim == 2:
        dist.squeeze_(0)
    elif ndim >= 4:
        dist = dist.unflatten(0, batch_shape)

    return dist


def surface_euclidean_distance_transform(images: Tensor,
                                         /,
                                         element_size: Optional[tuple[int | float, ...]] = None) -> Tensor:
    ndim = images.ndim
    if ndim == 2:
        images = images.unsqueeze(0)
    if ndim >= 4:
        batch_shape = images.shape[:-3]
        images = images.flatten(start_dim=0, end_dim=-4)

    is_vertex = is_surface_vertex(images)
    dist = cucim_edt(is_vertex, element_size=element_size)

    if ndim == 2:
        dist.squeeze_(0)
    elif ndim >= 4:
        dist = dist.unflatten(0, batch_shape)

    return dist
