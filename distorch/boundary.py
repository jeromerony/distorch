import functools
from typing import Optional

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from distorch.utils import batchify_args


@batchify_args('images')
def is_border_element(images: Tensor) -> Tensor:
    """
    For a batch of binary images of shape (b, h, w) or 3d volumes of shape (b, h, w, d), computes border
    pixels / voxels based on counting neighbors.

    Parameters
    ----------
    images : Tensor
        Boolean tensor where True values indicate the region for which to compute the border.

    Returns
    -------
    is_border : Tensor
        Boolean tensor indicating border pixels / voxels. Has the same shape as the input images.

    Examples
    --------
    >>> import torch
    >>> from distorch.boundary import is_border_element
    >>> img = torch.tensor([[ True,  True,  True, False,  True,  True],
    ...                     [ True,  True,  True, False, False,  True],
    ...                     [ True,  True,  True,  True, False, False],
    ...                     [ True,  True,  True, False, False, False],
    ...                     [False, False, False, False, False,  True]], dtype=torch.bool)
    >>> is_border_element(img)
    tensor([[ True,  True,  True, False,  True,  True],
            [ True, False,  True, False, False,  True],
            [ True, False, False,  True, False, False],
            [ True,  True,  True, False, False, False],
            [False, False, False, False, False,  True]])
    """
    device = images.device
    dtype = torch.uint8 if device.type == 'cpu' else torch.float16

    if images.ndim == 3:  # 2d images
        weight = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=dtype, device=images.device)
        num_neighbors = F.conv2d(images.to(dtype).unsqueeze(1),
                                 weight=weight.view(1, 1, 3, 3),
                                 stride=1, padding=1).squeeze(1)
        is_border = (num_neighbors < 4).logical_and_(images)

    elif images.ndim == 4:  # 3d volumes (..., h, w, d) : all leading dimensions are batch
        weight = torch.tensor([[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                               [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
                               [[0, 0, 0], [0, 1, 0], [0, 0, 0]]], dtype=dtype, device=images.device)
        num_neighbors = F.conv3d(images.to(dtype).unsqueeze(1),
                                 weight=weight.view(1, 1, 3, 3, 3),
                                 stride=1, padding=1).squeeze(1)
        is_border = (num_neighbors < 6).logical_and_(images)

    else:
        raise ValueError(f'Input should be Tensor with 3 or 4 dimensions: supplied {images.shape}')

    return is_border


@functools.lru_cache(maxsize=1)
def _vertices_elements_2d() -> list[np.ndarray]:
    # @formatter:off
    segments = np.array([[[-1, 0], [0, -1]],
                         [[-1, 0], [0,  1]],
                         [[ 1, 0], [0, -1]],
                         [[ 1, 0], [0,  1]]], dtype=np.int8)
    # @formatter:on
    vertices_segments = []
    arange = np.arange(1, 15, dtype=np.uint8)
    bits = np.unpackbits(arange[:, None], axis=1, count=4, bitorder='little').astype(bool)
    for b in bits:
        bit_segments = segments[b].reshape(-1, 2)
        # only keep segments appearing once: they belong to the boundary
        unique, counts = np.unique(bit_segments, axis=0, return_counts=True)
        surface_segments = unique[counts == 1]
        vertices_segments.append(surface_segments)
    return vertices_segments


@functools.lru_cache(maxsize=128)
def vertices_size_2d(element_size: Optional[tuple[float, float]] = None) -> Tensor:
    element_size = 0.5 if element_size is None else np.array(element_size) / 2
    vertices_segments = _vertices_elements_2d()
    sizes = np.zeros(16, dtype=np.float32)
    for i, elems in enumerate(vertices_segments):
        sizes[i + 1] = np.abs(elems * element_size).sum()  # all elements are 1d in 3d
    return torch.from_numpy(sizes)


@functools.lru_cache(maxsize=1)
def _vertices_elements_3d() -> list[np.ndarray]:
    # @formatter:off
    # represent each surface, which is a rectangle, by a vector representing its diagonal
    surfaces = np.array([[[-1, -1,  0], [ 0, -1, -1], [-1,  0, -1]],
                         [[-1, -1,  0], [ 0, -1,  1], [-1,  0,  1]],
                         [[-1,  0, -1], [ 0,  1, -1], [-1,  1,  0]],
                         [[-1,  0,  1], [-1,  1,  0], [ 0,  1,  1]],
                         [[ 1,  0, -1], [ 1, -1,  0], [ 0, -1, -1]],
                         [[ 1,  0,  1], [ 1, -1,  0], [ 0, -1,  1]],
                         [[ 0,  1, -1], [ 1,  0, -1], [ 1,  1,  0]],
                         [[ 1,  0,  1], [ 1,  1,  0], [ 0,  1,  1]]], dtype=np.int8)
    # @formatter:on
    vertices_elements = []
    arange = np.arange(1, 255, dtype=np.uint8)
    bits = np.unpackbits(arange[:, None], axis=1, bitorder='little').astype(bool)
    for b in bits:
        bit_surfaces = surfaces[b].reshape(-1, 3)
        # only keep surfaces appearing once: they belong to the boundary
        unique, counts = np.unique(bit_surfaces, axis=0, return_counts=True)
        exposed_surfaces = unique[counts == 1]
        vertices_elements.append(exposed_surfaces)
    return vertices_elements


@functools.lru_cache(maxsize=128)
def vertices_size_3d(element_size: Optional[tuple[float, float, float]] = None) -> Tensor:
    element_size = 0.5 if element_size is None else np.array(element_size) / 2
    vertices_elements = _vertices_elements_3d()
    sizes = np.zeros(256, dtype=np.float32)
    for i, elems in enumerate(vertices_elements):
        sized_elems = elems * element_size
        sized_elems[sized_elems == 0] = 1  # compute area of 2d rectangles in 3d => replace 0-thickness by 1
        sizes[i + 1] = np.abs(np.prod(sized_elems, axis=1)).sum()
    return torch.from_numpy(sizes)


@batchify_args('images')
def is_surface_vertex(images: Tensor,
                      return_size: bool = False,
                      element_size: Optional[tuple[float, ...]] = None) -> Tensor:
    """
    For a batch of binary images of shape (b, h, w) or 3d volumes of shape (b, h, w, d), computes surface vertices based
    on counting neighbors. For every pixel / voxel in the input, returns the grid of vertices forming the borders of
    these 2d or 3d volumes, where the value indicates whether the vertex in on the surface of a shape formed by True
    values in the inputs.

    For instance, given the following 2d image of size 4×4:
        [[0, 0, 0, 0],
         [0, 1, 1, 0]
         [0, 1, 1, 0]
         [0, 0, 0, 0]]
    The output will be the following grid of size 5×5
        [[0, 0, 0, 0, 0],
         [0, 1, 1, 1, 0]
         [0, 1, 0, 1, 0]
         [0, 1, 1, 1, 0]
         [0, 0, 0, 0, 0]]

    Parameters
    ----------
    images : Tensor
        Boolean tensor where True values indicate the region for which to compute the surface vertices.
    return_size : bool
        If True, returns the size of surface vertices instead of a binary mask.
    element_size : tuple of floats
        If provided, will adjust the size of surface vertices accordingly. Only used if `return_size=True`.

    Returns
    -------
    is_vertex : Tensor
        Boolean tensor indicating surface vertices. If `return_length` is true, returns int8 tensor.
        For any dimension of size d, the output has a corresponding dimension of size d+1.

    Examples
    --------
    >>> import torch
    >>> from distorch.boundary import is_surface_vertex
    >>> img = torch.tensor([[False, False, False, False,  True,  True],
    ...                     [False,  True,  True, False, False,  True],
    ...                     [False,  True,  True,  True, False, False],
    ...                     [False,  True,  True, False, False, False],
    ...                     [False, False, False, False, False,  True]], dtype=torch.bool)
    >>> is_surface_vertex(img)
    tensor([[False, False, False, False,  True,  True,  True],
            [False,  True,  True,  True,  True,  True,  True],
            [False,  True, False,  True,  True,  True,  True],
            [False,  True, False,  True,  True, False, False],
            [False,  True,  True,  True, False,  True,  True],
            [False, False, False, False, False,  True,  True]])

    """
    device = images.device
    dtype = torch.uint8 if device.type == 'cpu' else torch.float16
    images_converted = images.type(dtype)
    # enforce tuple of float to be hashable
    element_size = None if element_size is None else tuple(map(float, element_size))

    if images.ndim == 3:  # 2d images
        weight = 2 ** torch.arange(4, dtype=dtype, device=device)
        neighbors = F.conv2d(images_converted.unsqueeze(1),
                             weight=weight.reshape(1, 1, 2, 2),
                             stride=1, padding=1).squeeze(1).long()
        if return_size:
            vertices_size = vertices_size_2d(element_size=element_size).to(device, non_blocking=True)
            is_vertex = vertices_size[neighbors]
        else:
            is_vertex = (neighbors > 0).logical_and_(neighbors < 15)

    elif images.ndim == 4:  # 3d volumes (..., h, w, d) : all leading dimensions are batch
        weight = 2 ** torch.arange(8, dtype=dtype, device=device)
        neighbors = F.conv3d(images_converted.unsqueeze(1),
                             weight=weight.reshape(1, 1, 2, 2, 2),
                             stride=1, padding=1).squeeze(1).long()
        if return_size:
            vertices_size = vertices_size_3d(element_size=element_size).to(device, non_blocking=True)
            is_vertex = vertices_size[neighbors]
        else:
            is_vertex = (neighbors > 0).logical_and_(neighbors < 255)

    else:
        raise ValueError(f'Input should be Tensor with 3 or 4 dimensions: supplied {images.shape}')

    return is_vertex
