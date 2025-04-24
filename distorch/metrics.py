import dataclasses
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from distorch.boundary import is_border_element, is_surface_vertex
from distorch.min_pairwise_distance import minimum_distances
from distorch.utils import batchify_input_output, generate_coordinates, zero_padded_nonnegative_quantile


@dataclass
class DistanceMetrics:
    Hausdorff: Tensor
    Hausdorff95_1_to_2: Tensor
    Hausdorff95_2_to_1: Tensor
    AverageSurfaceDistance_1_to_2: Tensor
    AverageSurfaceDistance_2_to_1: Tensor
    AverageSymmetricSurfaceDistance: Tensor


@batchify_input_output
def set_metrics(set1: Tensor,
                set2: Tensor,
                /,
                element_size: Optional[tuple[int | float, ...]] = None) -> DistanceMetrics:
    assert set1.shape == set2.shape
    assert set1.dtype == torch.bool
    assert set2.dtype == torch.bool

    coords_shape = set1.shape[1:]
    coords = generate_coordinates(coords_shape, device=set1.device, element_size=element_size)

    zero, nan = set1.new_tensor(0, dtype=torch.float), set1.new_tensor(float('nan'), dtype=torch.float)
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

        dist_1_to_2 = minimum_distances(elem_1_not_2, elem_2)
        dist_2_to_1 = minimum_distances(elem_2_not_1, elem_1)

        metrics['Hausdorff'].append(torch.maximum(dist_1_to_2.max(), dist_2_to_1.max()))
        metrics['Hausdorff95_1_to_2'].append(zero_padded_nonnegative_quantile(dist_1_to_2, q=0.95, n=n1))
        metrics['Hausdorff95_2_to_1'].append(zero_padded_nonnegative_quantile(dist_2_to_1, q=0.95, n=n2))

        sum_dist_1, sum_dist_2 = dist_1_to_2.sum(), dist_2_to_1.sum()
        metrics['AverageSurfaceDistance_1_to_2'].append(sum_dist_1 / n1)
        metrics['AverageSurfaceDistance_2_to_1'].append(sum_dist_2 / n2)
        metrics['AverageSymmetricSurfaceDistance'].append((sum_dist_1 + sum_dist_2) / (n1 + n2))

    metrics = {k: torch.stack(v, dim=0) for k, v in metrics.items()}
    return DistanceMetrics(**metrics)


def boundary_metrics(images1: Tensor, images2: Tensor, /, **kwargs) -> DistanceMetrics:
    """
    Computes metrics between the boundaries of two sets. These metrics include the Hausdorff distance and it 95%
    variant, the average surface distances and their symmetric variant.
    This function uses the vertices of the elements (i.e. pixels, voxels) instead of the center of the element.
    For instance, an isolated pixel has 4 edges and 4 vertices, with an area of 1.
    This implementation uses grid-aligned, regularly spaced vertices to represent boundaries. Therefore, a straight line
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


def pixel_center_metrics(images1: Tensor, images2: Tensor, /, **kwargs) -> DistanceMetrics:
    """
    Computes distance between batches of images (or 3d volumes). The images should be binary, where True
    indicates that an element (i.e. pixel/voxel) belongs to the set for which we want to compute the Hausdorff distance.
    This function uses the center of elements (i.e. pixels, voxels) to represent them. For instance, an isolated pixel
    is represented by one point. This is the representation commonly used in libraries such as Monai or MedPy.

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
