import torch
from torch import Tensor

from distorch.distance_transform import euclidean_distance_transform


def expand_labels(label_images: Tensor,
                  /,
                  distance: int | float = 1,
                  background_label: int = 0) -> Tensor:
    # Adapted from `skimage.segmentation.expand_labels`
    if distance <= 0:
        return label_images

    ndim = label_images.ndim
    if ndim == 2:
        label_images = label_images.unsqueeze(0)
    if ndim >= 4:
        batch_shape = label_images.shape[:-3]
        label_images = label_images.flatten(start_dim=0, end_dim=-4)

    distances, nearest_label_index = euclidean_distance_transform(label_images != background_label, return_indices=True)
    nearest_labels = label_images.flatten(start_dim=1).gather(dim=1, index=nearest_label_index.flatten(start_dim=1))
    nearest_labels = nearest_labels.unflatten(dim=1, sizes=label_images.shape[1:])
    dilate_mask = distances <= distance
    labels_out = torch.where(dilate_mask, nearest_labels, label_images)

    if ndim == 2:
        labels_out.squeeze_(0)
    elif ndim >= 4:
        labels_out = labels_out.unflatten(0, batch_shape)

    return labels_out
