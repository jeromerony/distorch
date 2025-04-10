from collections import defaultdict
from typing import Optional

import torch
from torch import Tensor

from distorch import use_pykeops
from distorch.boundary import is_border_element, is_surface_vertex
from distorch.distance_transform import euclidean_distance_transform
from distorch.utils import generate_coordinates

if use_pykeops:
    from pykeops.torch import LazyTensor


def hausdorff(set1: Tensor, set2: Tensor,
              /,
              element_size: Optional[tuple[int | float, ...]] = None) -> dict[str, Tensor]:
    assert set1.shape == set2.shape
    assert set1.dtype == torch.bool
    assert set2.dtype == torch.bool

    ndim = set1.ndim
    if ndim == 2:
        set1, set2 = set1.unsqueeze(0), set2.unsqueeze(0)
    elif ndim >= 4:
        batch_shape = set1.shape[:-3]
        set1, set2 = set1.flatten(start_dim=0, end_dim=-4), set2.flatten(start_dim=0, end_dim=-4)

    edt1: Tensor = euclidean_distance_transform(set1, element_size=element_size)
    edt2: Tensor = euclidean_distance_transform(set2, element_size=element_size)
    h = edt1.sub_(edt2).abs_().flatten(start_dim=1).amax(dim=1)

    if ndim == 2:
        h.squeeze_(0)
    elif ndim >= 4:
        h = h.unflatten(0, batch_shape)

    return h


def set_metrics(set1: Tensor, set2: Tensor,
                /,
                quantile: float | Tensor = 0.95,
                element_size: Optional[tuple[int | float, ...]] = None) -> dict[str, Tensor]:
    assert set1.shape == set2.shape
    assert set1.dtype == torch.bool
    assert set2.dtype == torch.bool

    ndim = set1.ndim
    if ndim == 2:
        set1, set2 = set1.unsqueeze(0), set2.unsqueeze(0)
    elif ndim >= 4:
        batch_shape = set1.shape[:-3]
        set1, set2 = set1.flatten(start_dim=0, end_dim=-4), set2.flatten(start_dim=0, end_dim=-4)

    coords_shape = set1.shape[1:]
    coords_ndim = len(coords_shape)
    coords = generate_coordinates(coords_shape, device=set1.device, element_size=element_size)

    metrics = defaultdict(list)
    for s1, s2 in zip(set1, set2):
        elem_1 = coords[s1].view(-1, 1, coords_ndim)
        elem_2 = coords[s2].view(1, -1, coords_ndim)

        if use_pykeops:
            pairwise_distances = (LazyTensor(elem_1) - LazyTensor(elem_2)).norm2()
            dist_1_to_2 = pairwise_distances.min(dim=1)
            dist_2_to_1 = pairwise_distances.min(dim=0)

        else:
            pairwise_distances = (elem_1 - elem_2).norm(p=2, dim=2)
            dist_1_to_2 = pairwise_distances.amin(dim=1)
            dist_2_to_1 = pairwise_distances.amin(dim=0)

        metrics['hausdorff'].append(torch.maximum(dist_1_to_2.max(), dist_2_to_1.max()))
        # metrics['average_distance_1_to_2'].append(dist_1_to_2.mean())
        # metrics['average_distance_2_to_1'].append(dist_2_to_1.mean())
        # metrics[f'distance_{quantile:.0%}_1_to_2'].append(torch.quantile(dist_1_to_2, q=quantile))
        # metrics[f'distance_{quantile:.0%}_2_to_1'].append(torch.quantile(dist_2_to_1, q=quantile))

    metrics = {k: torch.stack(v, dim=0) for k, v in metrics.items()}

    if ndim == 2:
        metrics = {k: v.squeeze(0) for k, v in metrics.items()}
    elif ndim >= 4:
        metrics = {k: v.unflatten(0, batch_shape) for k, v in metrics.items()}

    return metrics


def border_metrics(images1: Tensor,
                   images2: Tensor,
                   /,
                   quantile: float | Tensor = 0.95,
                   element_size: Optional[tuple[int | float, ...]] = None) -> dict[str, Tensor]:
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
    set1, set2 = is_border_element(images1), is_border_element(images2)
    metrics = set_metrics(set1, set2, quantile=quantile, element_size=element_size)
    metrics = {f'border_{k}': v for k, v in metrics.items()}
    return metrics


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
    set1, set2 = is_surface_vertex(images1), is_surface_vertex(images2)
    metrics = set_metrics(set1, set2, quantile=quantile, element_size=element_size)
    metrics = {f'surface_{k}': v for k, v in metrics.items()}
    return metrics
