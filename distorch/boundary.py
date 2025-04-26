import torch
from torch import Tensor
from torch.nn import functional as F

from distorch.utils import batchify_input_output


@batchify_input_output
def is_border_element(images: Tensor, /) -> Tensor:
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


@batchify_input_output
def is_surface_vertex(images: Tensor, /, return_length: bool = False) -> Tensor:
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
    return_length : bool
        If True, returns the "length" of surface vertices instead of a binary mask.

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
    images_converted = images.type(dtype, non_blocking=True)

    if images.ndim == 3:  # 2d images

        diag_weight = images_converted.new_tensor([[[1, 0],
                                                    [0, 1]],
                                                   [[0, 1],
                                                    [1, 0]]])
        diag_conv = F.conv2d(images_converted.unsqueeze(1), weight=diag_weight.unsqueeze(1), stride=1, padding=1)
        neighbors = diag_conv.sum(dim=1)
        is_vertex = (neighbors > 0).logical_and_(neighbors < 4)
        if return_length:
            is_diag_vertex = ((diag_conv == 2) & torch.flip(diag_conv == 0, dims=(1,))).any(dim=1)
            is_vertex = is_vertex.type(torch.int8).add_(is_diag_vertex)

    elif images.ndim == 4:  # 3d volumes (..., h, w, d) : all leading dimensions are batch
        weight = images_converted.new_ones(())
        neighbors = F.conv3d(images_converted.unsqueeze(1),
                             weight=weight.expand(1, 1, 2, 2, 2),
                             stride=1, padding=1).squeeze(1)
        is_vertex = (neighbors > 0).logical_and_(neighbors < 8)

    else:
        raise ValueError(f'Input should be Tensor with 3 or 4 dimensions: supplied {images.shape}')

    return is_vertex


if __name__ == '__main__':
    A = torch.tensor([[1, 0, 0, 1],
                      [0, 1, 1, 0],
                      [0, 1, 1, 0],
                      [0, 0, 0, 0]], dtype=torch.bool)

    print(is_surface_vertex(A).int())
    print(is_surface_vertex(A, return_length=True).int())
