import dataclasses
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

import distorch
from distorch.boundary import is_border_element, is_surface_vertex
from distorch.min_pairwise_distance.cpu import min_sqdist_numba
from distorch.utils import generate_coordinates

if distorch.use_triton:
    from distorch.min_pairwise_distance.gpu import min_sqdist_triton

if distorch.use_pykeops:
    from pykeops.torch import Vi, Vj
else:
    import warnings

    warnings.warn(
        'PyKeops or Triton could not be imported, this will result in high memory usage and/or out-of-memory crash.'
    )


@dataclass
class DistanceMetrics:
    Hausdorff: Tensor
    Hausdorff95_1_to_2: Tensor
    Hausdorff95_2_to_1: Tensor
    AverageSurfaceDistance_1_to_2: Tensor
    AverageSurfaceDistance_2_to_1: Tensor
    AverageSymmetricSurfaceDistance: Tensor


def set_metrics(set1: Tensor,
                set2: Tensor,
                /,
                element_size: Optional[tuple[int | float, ...]] = None) -> DistanceMetrics:
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
    coords = generate_coordinates(coords_shape, device=set1.device, element_size=element_size)

    zero = torch.tensor(0., device=set1.device)
    nan = torch.tensor(float('nan'), device=set1.device)
    metrics = {f.name: [] for f in dataclasses.fields(DistanceMetrics)}
    for s1, s2 in zip(set1, set2):
        elem_1 = coords[s1]
        elem_2 = coords[s2]
        n1, n2 = len(elem_1), len(elem_2)

        if n1 < 1 or n2 < 1:  # one set is empty
            [m.append(nan) for m in metrics.values()]
            continue
        elif torch.equal(elem_1, elem_2):  # both are non-empty but equal
            [m.append(zero) for m in metrics.values()]
            continue

        elem_1_not_2 = coords[s2.logical_not().logical_and_(s1)]
        elem_2_not_1 = coords[s1.logical_not().logical_and_(s2)]

        if set1.device.type == 'cpu':
            dist_1_to_2 = min_sqdist_numba(elem_1_not_2, elem_2)
            dist_2_to_1 = min_sqdist_numba(elem_2_not_1, elem_1)
            dist_1_to_2.sqrt_(), dist_2_to_1.sqrt_()

        elif distorch.use_pykeops:
            dist_1_to_2 = Vi(elem_1_not_2).sqdist(Vj(elem_2)).min(dim=1).squeeze(1).sqrt_()
            dist_2_to_1 = Vi(elem_2_not_1).sqdist(Vj(elem_1)).min(dim=1).squeeze(1).sqrt_()

        elif distorch.use_triton:
            dist_1_to_2 = min_sqdist_triton(elem_1_not_2, elem_2).sqrt_()
            dist_2_to_1 = min_sqdist_triton(elem_2_not_1, elem_1).sqrt_()

        else:  # defaults to native
            dist_1_to_2 = torch.cdist(elem_1_not_2, elem_2).amin(dim=1)
            dist_2_to_1 = torch.cdist(elem_2_not_1, elem_1).amin(dim=1)

        metrics['Hausdorff'].append(torch.maximum(dist_1_to_2.max(), dist_2_to_1.max()))

        hd95_1_to_2 = torch.quantile(dist_1_to_2, q=1 - (1 - 0.95) * n1 / elem_1_not_2.size(0))
        hd95_2_to_1 = torch.quantile(dist_2_to_1, q=1 - (1 - 0.95) * n2 / elem_2_not_1.size(0))
        metrics['Hausdorff95_1_to_2'].append(hd95_1_to_2)
        metrics['Hausdorff95_2_to_1'].append(hd95_2_to_1)

        sum_dist_1, sum_dist_2 = dist_1_to_2.sum(), dist_2_to_1.sum()
        metrics['AverageSurfaceDistance_1_to_2'].append(sum_dist_1 / n1)
        metrics['AverageSurfaceDistance_2_to_1'].append(sum_dist_2 / n2)
        metrics['AverageSymmetricSurfaceDistance'].append((sum_dist_1 + sum_dist_2) / (n1 + n2))

    metrics = {k: torch.stack(v, dim=0) for k, v in metrics.items()}
    if ndim == 2:
        metrics = {k: v.squeeze(0) for k, v in metrics.items()}
    elif ndim >= 4:
        metrics = {k: v.unflatten(0, batch_shape) for k, v in metrics.items()}

    return DistanceMetrics(**metrics)


def border_metrics(images1: Tensor, images2: Tensor, /, **kwargs) -> DistanceMetrics:
    """
    Computes the Hausdorff distances between batches of images (or 3d volumes). The images should be binary, where True
    indicates that an element (i.e. pixel/voxel) belongs to the set for which we want to compute the Hausdorff distance.

    Parameters
    ----------
    images1 : Tensor
        Boolean tensor indicating the membership to the first set.
    images2 : Tensor
        Boolean tensor indicating the membership to the second set.

    Returns
    -------
    metrics : dict[str, Tensor]
        Dictionary of metrics where each entry correspond to a metric for all the element in the batch.

    """
    set1, set2 = is_border_element(images1), is_border_element(images2)
    return set_metrics(set1, set2, **kwargs)


def surface_metrics(images1: Tensor, images2: Tensor, /, **kwargs) -> DistanceMetrics:
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
    images2 : Tensor
        Boolean tensor indicating the membership to the second set.

    Returns
    -------
    metrics : dict[str, Tensor]
        Dictionary of metrics where each entry correspond to a metric for all the element in the batch.

    """
    set1, set2 = is_surface_vertex(images1), is_surface_vertex(images2)
    return set_metrics(set1, set2, **kwargs)
