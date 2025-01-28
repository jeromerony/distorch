import torch
from torch import Tensor
from torch.nn import functional as F


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
            [ True, False,  True,  True, False, False],
            [ True,  True,  True, False, False, False],
            [False, False, False, False, False,  True]])
    """
    device = images.device
    dtype = torch.uint8 if device.type == 'cpu' else torch.float16
    weight = torch.ones((), dtype=dtype, device=images.device)

    if images.ndim in (2, 3):  # 2d images
        num_neighbors = F.conv2d(images.to(dtype).reshape(-1, 1, *images.shape[-2:]),
                                 weight=weight.expand(1, 1, 3, 3),
                                 stride=1, padding=1).reshape_as(images)
        is_border = (num_neighbors < 9).logical_and_(images)

    elif images.ndim > 4:  # 3d volumes (..., h, w, d) : all leading dimensions are batch
        num_neighbors = F.conv3d(images.to(dtype).flatten(start_dim=0, end_dim=-4).unsqueeze(1),
                                 weight=weight.expand(1, 1, 3, 3, 3),
                                 stride=1, padding=1).reshape_as(images)
        is_border = (num_neighbors < 27).logical_and_(images)

    else:
        raise ValueError(f'Input should be Tensor with at least 2 dimensions: supplied {images.shape}')

    return is_border


def is_surface_vertex(images: Tensor) -> Tensor:
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

    Returns
    -------
    is_vertex : Tensor
        Boolean tensor indicating surface vertices.
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
    weight = torch.ones((), dtype=dtype, device=images.device)

    if images.ndim in (2, 3):  # 2d images
        neighbors = F.conv2d(images.to(dtype).reshape(-1, 1, *images.shape[-2:]),
                             weight=weight.expand(1, 1, 2, 2),
                             stride=1, padding=1).squeeze(1)
        if images.ndim == 2:
            neighbors.squeeze_(0)
        is_vertex = (neighbors > 0).logical_and_(neighbors < 4)

    elif images.ndim > 4:  # 3d volumes (..., h, w, d) : all leading dimensions are batch
        neighbors = F.conv3d(images.to(dtype).flatten(start_dim=0, end_dim=-4).unsqueeze(1),
                             weight=weight.expand(1, 1, 2, 2, 2),
                             stride=1, padding=1).squeeze(1)
        is_vertex = (neighbors > 0).logical_and_(neighbors < 8).unflatten(0, images.shape[:-3])

    else:
        raise ValueError(f'Input should be Tensor with at least 2 dimensions: supplied {images.shape}')

    return is_vertex
