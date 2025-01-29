from collections import defaultdict
from typing import Optional

import torch
from torch import Tensor

from distorch import use_pykeops
from distorch.boundary import is_surface_vertex
from distorch.distance_transform import euclidean_distance_transform
from distorch.utils import generate_coordinates

if use_pykeops:
    from pykeops.torch import LazyTensor


def hausdorff(images1: Tensor,
              images2: Tensor,
              /,
              element_size: Optional[tuple[int | float, ...]] = None) -> Tensor:
    """
    Computes the Hausdorff distances between batches of images (or 3d volumes). The images should be binary, where True
    indicates that an element (i.e. pixel/voxel) belongs to the set for which we want to compute the Hausdorff distance.

    Parameters
    ----------
    images1 : Tensor
        Boolean tensor indicating the membership to the first set.
    images2
        Boolean tensor indicating the membership to the second set.
    element_size : tuple of ints or floats
        Size of a single spatial element (pixel / voxel) along each dimension. Defaults to 1 for every dimension.

    Returns
    -------
    h : Tensor
        Hausdorff distances between the batches of first and second sets.

    """
    assert images1.shape == images2.shape
    ndim = images1.ndim
    if ndim == 2:
        images1, images2 = images1.unsqueeze(0), images2.unsqueeze(0)
    elif ndim >= 4:
        batch_shape = images1.shape[:-3]
        images1, images2 = images1.flatten(start_dim=0, end_dim=-4), images2.flatten(start_dim=0, end_dim=-4)

    edt1 = euclidean_distance_transform(images1, element_size=element_size)
    edt2 = euclidean_distance_transform(images2, element_size=element_size)
    h = edt1.sub_(edt2).abs_().flatten(start_dim=1).amax(dim=1)

    if ndim == 2:
        h.squeeze_(0)
    elif ndim >= 4:
        h = h.unflatten(0, batch_shape)

    return h


def surface_metrics(images1: Tensor,
                    images2: Tensor,
                    /,
                    quantile: float | Tensor = 0.95,
                    element_size: Optional[tuple[int | float, ...]] = None) -> dict[str, Tensor]:
    """
    Computes metrics between the surfaces of two sets. These metrics include the surface Hausdorff distance, the
    directed average surface distances and the quantile of the directed surface distances (also called Hausdorff 95%).
    This function computes the distances between the true surfaces of the sets instead of using the center of the
    element. For instance, an isolated pixel has 4 edges and 4 vertices, with an area of 1.
    This implementation uses grid-aligned, regularly spaced vertices to represent surfaces. Therefore, a straight line
    of length 5 (in pixel space) is formed by 6 vertices.

    Parameters
    ----------
    images1 : Tensor
        Boolean tensor indicating the membership to the first set.
    images2
        Boolean tensor indicating the membership to the second set.
    quantile : float or Tensor
        Argument passed to the `torch.quantile` function.
    element_size : tuple of ints or floats
        Size of a single spatial element (pixel / voxel) along each dimension. Defaults to 1 for every dimension.

    Returns
    -------
    metrics : dict[str, Tensor]
        Dictionary of metrics where each entry correspond to a metric for all the element in the batch.

    """
    assert images1.shape == images2.shape
    device = images1.device

    ndim = images1.ndim
    if ndim == 2:
        images1, images2 = images1.unsqueeze(0), images2.unsqueeze(0)
    elif ndim >= 4:
        batch_shape = images1.shape[:-3]
        images1, images2 = images1.flatten(start_dim=0, end_dim=-4), images2.flatten(start_dim=0, end_dim=-4)

    is_vertex_1 = is_surface_vertex(images1)
    is_vertex_2 = is_surface_vertex(images2)

    coords_shape = [s + 1 for s in images1.shape[1:]]
    coords_ndim = len(coords_shape)
    coords = generate_coordinates(coords_shape, device=device, element_size=element_size)

    metrics = defaultdict(list)
    for is_v_1, is_v_2 in zip(is_vertex_1, is_vertex_2):
        surface_vertices_1 = coords[is_v_1].view(-1, 1, coords_ndim)
        surface_vertices_2 = coords[is_v_2].view(1, -1, coords_ndim)

        if use_pykeops:
            pairwise_distances = (LazyTensor(surface_vertices_1) - LazyTensor(surface_vertices_2)).norm2()
            dist_1_to_2 = pairwise_distances.min(dim=1)
            dist_2_to_1 = pairwise_distances.min(dim=0)

        else:
            pairwise_distances = (surface_vertices_1 - surface_vertices_2).norm(p=2, dim=2)
            dist_1_to_2 = pairwise_distances.amin(dim=1)
            dist_2_to_1 = pairwise_distances.amin(dim=0)

        metrics['hausdorff'].append(torch.maximum(dist_1_to_2.max(), dist_2_to_1.max()))
        metrics['average_surface_distance_1_to_2'].append(dist_1_to_2.mean())
        metrics['average_surface_distance_2_to_1'].append(dist_2_to_1.mean())
        metrics[f'hausdorff_1_to_2_{quantile:.0%}'].append(torch.quantile(dist_1_to_2, q=quantile))
        metrics[f'hausdorff_2_to_1_{quantile:.0%}'].append(torch.quantile(dist_2_to_1, q=quantile))

    metrics = {k: torch.stack(v, dim=0) for k, v in metrics.items()}

    if ndim == 2:
        metrics = {k: v.squeeze(0) for k, v in metrics.items()}
    elif ndim >= 4:
        metrics = {k: v.unflatten(0, batch_shape) for k, v in metrics.items()}

    return metrics
