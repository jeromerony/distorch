from pprint import pprint

import nibabel
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.nn import functional as F
from torch.utils.benchmark import Timer

from distorch.distance_transform import surface_euclidean_distance_transform
from distorch.metrics import surface_metrics

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    h, w = 20, 20
    images = torch.zeros(2, h, w, device=device, dtype=torch.bool)
    images[:, 4:16, 4:16] = True
    images[0, 8:12, 4:10] = False

    s_d = surface_euclidean_distance_transform(images)

    fig, axes = plt.subplots(2, 2, layout='constrained')
    imshow_kwargs = dict(vmin=0, vmax=1, cmap='gray', origin='lower', extent=(0, h, 0, w))
    axes[0, 0].imshow(images[0].cpu(), **imshow_kwargs)
    axes[0, 1].imshow(images[1].cpu(), **imshow_kwargs)

    vmax = s_d.max().item()
    axes[1, 0].imshow(s_d[0].cpu(), vmin=0, vmax=vmax, origin='lower')
    axes[1, 1].imshow(s_d[1].cpu(), vmin=0, vmax=vmax, origin='lower')

    ticks = np.arange(0, 21, step=4)
    for ax in axes.ravel():
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

    plt.show(block=True)

    gt = torch.from_numpy(np.asarray(nibabel.load('data/9996098_R_segmentation.nii.gz').dataobj)).to(dtype=torch.long)
    prediction = torch.from_numpy(np.asarray(nibabel.load('data/9996098.nii.gz').dataobj)).to(dtype=torch.long)
    k = 5

    gt_onehot = F.one_hot(gt.to(device), num_classes=k).movedim(-1, 0).bool().contiguous()
    prediction_onehot = F.one_hot(prediction.to(device), num_classes=k).movedim(-1, 0).bool().contiguous()

    print(gt_onehot.shape, gt_onehot.dtype, prediction_onehot.shape, prediction_onehot.dtype)

    out = surface_metrics(prediction_onehot, gt_onehot)
    pprint(out)

    timer = Timer(stmt='surface_metrics(prediction_onehot, gt_onehot)',
                  setup='from distorch.metrics import surface_metrics', num_threads=8,
                  globals={'prediction_onehot': prediction_onehot, 'gt_onehot': gt_onehot})
    print(timer.blocked_autorange(min_run_time=10))
