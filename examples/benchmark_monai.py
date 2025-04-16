from pprint import pprint

import nibabel
import numpy as np
import torch
from cupyx.profiler import benchmark
from monai.metrics import compute_average_surface_distance, compute_hausdorff_distance
from torch.nn import functional as F

from distorch.metrics import surface_metrics

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    # gt = torch.from_numpy(np.asarray(nibabel.load('data/9996098_R_segmentation.nii.gz').dataobj)).to(dtype=torch.long)
    # prediction = torch.from_numpy(np.asarray(nibabel.load('data/9996098.nii.gz').dataobj)).to(dtype=torch.long)
    gt = torch.from_numpy(np.asarray(nibabel.load('data/patient02_gt.nii.gz').dataobj)).to(dtype=torch.long)
    prediction = torch.from_numpy(np.asarray(nibabel.load('data/patient02_prediction.nii.gz').dataobj)).to(
        dtype=torch.long)
    k = 5

    gt_onehot = F.one_hot(gt.to(device), num_classes=k).movedim(-1, 0).bool().contiguous()
    prediction_onehot = F.one_hot(prediction.to(device), num_classes=k).movedim(-1, 0).bool().contiguous()
    # remove background class
    gt_onehot, prediction_onehot = gt_onehot[1:], prediction_onehot[1:]

    print(gt_onehot.shape, gt_onehot.dtype, prediction_onehot.shape, prediction_onehot.dtype)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    surf_metrics = surface_metrics(prediction_onehot, gt_onehot)
    pprint(surf_metrics)
    print(f'pyKeops max memory: {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB')
    bench = benchmark(surface_metrics, (prediction_onehot, gt_onehot), n_repeat=5)
    if device.type == 'cuda':
        print(f'pyKeops GPU: {np.mean(bench.gpu_times):.4g} ± {np.std(bench.gpu_times):.4g}')
    else:
        print(f'pyKeops CPU: {np.mean(bench.cpu_times):.4g} ± {np.std(bench.cpu_times):.4g}')

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    monai_hausdorff = compute_hausdorff_distance(prediction_onehot.unsqueeze(0), gt_onehot.unsqueeze(0),
                                                 include_background=True)
    pprint(monai_hausdorff)
    print(f'Monai max memory: {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB')
    bench = benchmark(compute_hausdorff_distance, (prediction_onehot.unsqueeze(0), gt_onehot.unsqueeze(0)),
                      n_repeat=5)
    if device.type == 'cuda':
        print(f'Monai GPU: {np.mean(bench.gpu_times):.4g} ± {np.std(bench.gpu_times):.4g}')
    else:
        print(f'Monai CPU: {np.mean(bench.cpu_times):.4g} ± {np.std(bench.cpu_times):.4g}')
