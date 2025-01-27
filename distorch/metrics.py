from collections import defaultdict

import torch
from torch import Tensor

from distorch.distance_transform import euclidean_distance_transform
from distorch.surface import is_surface_vertex

use_pykeops = True
try:
    from pykeops.torch import LazyTensor
except:
    import warnings

    warnings.warn('PyKeops could not be imported, this will result in high memory usage.')
    use_pykeops = False


def hausdorff(images1: Tensor, images2: Tensor) -> Tensor:
    """
    Computes the Hausdorff distances between batches of images (or 3d volumes). The images should be binary, where True
    indicates that an element (i.e. pixel/voxel) belongs to the set for which we want to compute the Hausdorff distance.

    Parameters
    ----------
    images1 : Tensor
        Boolean tensor indicating the membership to the first set.
    images2
        Boolean tensor indicating the membership to the second set.

    Returns
    -------
    h : Tensor
        Hausdorff distances between the batches of first and second sets.

    """
    edt1 = euclidean_distance_transform(images1)
    edt2 = euclidean_distance_transform(images2)
    h = edt1.sub_(edt2).abs_().flatten(start_dim=1).amax(dim=1)
    return h


def surface_metrics(images1: Tensor,
                    images2: Tensor,
                    /,
                    quantile: float | Tensor = 0.95) -> dict[str, Tensor]:
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

    Returns
    -------
    metrics : dict[str, Tensor]
        Dictionary of metrics where each entry correspond to a metric for all the element in the batch.

    """
    assert images1.shape == images2.shape
    device = images1.device
    is_vertex_1 = is_surface_vertex(images1)
    is_vertex_2 = is_surface_vertex(images2)

    coords_shape = [s + 1 for s in images1.shape[1:]]
    coords_ndim = len(coords_shape)
    aranges = [torch.arange(s, device=device, dtype=torch.float) for s in coords_shape]
    coords = torch.stack(torch.meshgrid(*aranges, indexing='ij'), dim=-1)

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
    return metrics
