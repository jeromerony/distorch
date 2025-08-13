import dataclasses
from dataclasses import dataclass
from typing import Optional

import torch
from torch import SymInt, Tensor

from distorch.boundary import is_border_element, is_surface_vertex
from distorch.min_pairwise_distance import minimum_distances
from distorch.utils import batchify_args, zero_padded_nonnegative_quantile


@dataclass
class DistanceMetrics:
    Hausdorff: Tensor
    Hausdorff95_1_to_2: Tensor
    Hausdorff95_2_to_1: Tensor
    AverageSurfaceDistance_1_to_2: Tensor
    AverageSurfaceDistance_2_to_1: Tensor
    AverageSymmetricSurfaceDistance: Tensor
    NormalizedSurfaceDistance_1_to_2: Tensor
    NormalizedSurfaceDistance_2_to_1: Tensor
    NormalizedSymmetricSurfaceDistance: Tensor


def reframe(masks: tuple[Tensor, Tensor]) -> list[Tensor]:
    ndim = masks[0].ndim
    dims = set(range(ndim))
    for dim in range(ndim):
        other_dims = tuple(dims - {dim})
        non_empty = masks[0].any(dim=other_dims).logical_or_(masks[1].any(dim=other_dims))
        arange = torch.arange(masks[0].size(dim), device=masks[0].device)

        left = torch.where(non_empty, arange, masks[0].size(dim) - 1).amin()
        right = torch.where(non_empty, arange, 0).amax().add_(1)
        length: SymInt = (right - left).clamp_(min=0)

        masks = [mask.narrow(dim=dim, start=left, length=length) for mask in masks]

    return masks


def mask_to_coords(mask: Tensor, element_size: Optional[tuple[int | float, ...]] = None) -> Tensor:
    coords = [c.type(torch.float) for c in torch.where(mask)]
    if element_size is not None:
        torch._foreach_mul_(coords, element_size)
    return torch.stack(coords, dim=1)


@batchify_args('set1', 'set2')
def set_metrics(set1: Tensor,
                set2: Tensor,
                element_size: Optional[tuple[int | float, ...]] = None,
                distance_threshold: float = 1) -> DistanceMetrics:
    assert set1.shape == set2.shape
    assert set1.dtype == set2.dtype

    zero, nan = set1.new_zeros((), dtype=torch.float), set1.new_full((), float('nan'), dtype=torch.float)
    metrics = {f.name: [] for f in dataclasses.fields(DistanceMetrics)}
    for sizes_1, sizes_2 in zip(set1, set2):
        sizes_1, sizes_2 = reframe((sizes_1, sizes_2))

        if set1.dtype == torch.bool and set2.dtype == torch.bool:
            s1, s2 = sizes_1, sizes_2
        else:
            s1, s2 = sizes_1 > 0, sizes_2 > 0

        elem_1 = mask_to_coords(s1, element_size=element_size)
        elem_2 = mask_to_coords(s2, element_size=element_size)
        n1, n2 = len(elem_1), len(elem_2)

        if n1 < 1 or n2 < 1:  # one set is empty
            [m.append(nan) for m in metrics.values()]
            continue
        elif torch.equal(elem_1, elem_2):  # both are non-empty but equal
            [m.append(zero) for m in metrics.values()]
            continue

        mask_1_not_2 = s2.logical_not().logical_and_(s1)
        elem_1_not_2 = mask_to_coords(mask_1_not_2, element_size=element_size)

        mask_2_not_1 = s1.logical_not().logical_and_(s2)
        elem_2_not_1 = mask_to_coords(mask_2_not_1, element_size=element_size)

        dist_1_to_2 = minimum_distances(elem_1_not_2, elem_2)
        dist_2_to_1 = minimum_distances(elem_2_not_1, elem_1)

        metrics['Hausdorff'].append(torch.maximum(dist_1_to_2.max(), dist_2_to_1.max()))
        # TODO: take the elements size into account in quantile
        metrics['Hausdorff95_1_to_2'].append(zero_padded_nonnegative_quantile(dist_1_to_2, q=0.95, n=n1))
        metrics['Hausdorff95_2_to_1'].append(zero_padded_nonnegative_quantile(dist_2_to_1, q=0.95, n=n2))

        sizes_1_not_2, sizes_2_not_1 = sizes_1[mask_1_not_2], sizes_2[mask_2_not_1]
        total_size_1, total_size_2 = sizes_1.sum(), sizes_2.sum()

        weighted_sum_dist_1 = (dist_1_to_2 * sizes_1_not_2).sum()
        weighted_sum_dist_2 = (dist_2_to_1 * sizes_2_not_1).sum()
        metrics['AverageSurfaceDistance_1_to_2'].append(weighted_sum_dist_1 / total_size_1)
        metrics['AverageSurfaceDistance_2_to_1'].append(weighted_sum_dist_2 / total_size_2)
        metrics['AverageSymmetricSurfaceDistance'].append(
            (weighted_sum_dist_1 + weighted_sum_dist_2) / (total_size_1 + total_size_2)
        )

        dist_1_to_2_larger = dist_1_to_2 > distance_threshold
        dist_2_to_1_larger = dist_2_to_1 > distance_threshold
        weighted_sum_smaller_1_to_2 = total_size_1 - (sizes_1_not_2 * dist_1_to_2_larger).sum()
        weighted_sum_smaller_2_to_1 = total_size_2 - (sizes_2_not_1 * dist_2_to_1_larger).sum()
        metrics['NormalizedSurfaceDistance_1_to_2'].append(weighted_sum_smaller_1_to_2 / total_size_1)
        metrics['NormalizedSurfaceDistance_2_to_1'].append(weighted_sum_smaller_2_to_1 / total_size_2)
        metrics['NormalizedSymmetricSurfaceDistance'].append(
            (weighted_sum_smaller_1_to_2 + weighted_sum_smaller_2_to_1) / (total_size_1 + total_size_2)
        )

    metrics = {k: torch.stack(v, dim=0) for k, v in metrics.items()}
    return DistanceMetrics(**metrics)


def boundary_metrics(images1: Tensor,
                     images2: Tensor,
                     weight_by_size: bool = True,
                     element_size: Optional[tuple[int | float, ...]] = None,
                     **kwargs) -> DistanceMetrics:
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
    vertices_1 = is_surface_vertex(images1, element_size=element_size, return_size=weight_by_size)
    vertices_2 = is_surface_vertex(images2, element_size=element_size, return_size=weight_by_size)
    return set_metrics(vertices_1, vertices_2, element_size=element_size, **kwargs)


def pixel_center_metrics(images1: Tensor, images2: Tensor, **kwargs) -> DistanceMetrics:
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


@dataclass
class OverlapMetrics:
    Dice: Tensor
    Jaccard: Tensor
    ConfusionMatrix: Tensor
    PixelAccuracy: Tensor
    OverallPixelAccuracy: Tensor


@batchify_args('pred', 'ground_truth')
def overlap_metrics(pred: Tensor, ground_truth: Tensor, num_classes: int) -> OverlapMetrics:
    confusion_matrix = ground_truth.new_zeros(ground_truth.size(0), num_classes ** 2, dtype=torch.long)
    confusion_matrix.scatter_(
        dim=1, index=pred.add(ground_truth, alpha=num_classes).flatten(1), value=1, reduce='add'
    )  # batched bincount
    confusion_matrix = confusion_matrix.unflatten(dim=1, sizes=(num_classes, num_classes))
    class_TP = confusion_matrix.diagonal(dim1=1, dim2=2)
    class_gt = confusion_matrix.sum(dim=2)
    class_pred = confusion_matrix.sum(dim=1)
    dice_denominator = class_gt + class_pred
    class_dice = torch.where(dice_denominator > 0, 2 * class_TP / dice_denominator, float('nan'))
    jaccard_denominator = dice_denominator - class_TP
    class_jaccard = torch.where(jaccard_denominator > 0, class_TP / jaccard_denominator, float('nan'))
    pixel_accuracy = torch.where(class_gt > 0, class_TP / class_gt, float('nan'))
    overall_pixel_accuracy = class_TP.sum(dim=1) / confusion_matrix.sum(dim=(1, 2))
    return OverlapMetrics(
        Dice=class_dice,
        Jaccard=class_jaccard,
        ConfusionMatrix=confusion_matrix,
        PixelAccuracy=pixel_accuracy,
        OverallPixelAccuracy=overall_pixel_accuracy
    )
