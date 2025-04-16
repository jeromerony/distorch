from pprint import pprint

import nibabel
import numpy as np
import torch
from monai.metrics import compute_average_surface_distance, compute_hausdorff_distance
from torch.nn import functional as F

from distorch.metrics import surface_metrics, border_metrics

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    k = 5
    gt_nii = nibabel.load('data/9996098_R_segmentation.nii.gz')
    pred_nii = nibabel.load('data/9996098.nii.gz')
    # gt_nii = nibabel.load('data/patient02_gt.nii.gz')
    # pred_nii = nibabel.load('data/patient02_prediction.nii.gz')

    element_size = tuple(map(float, gt_nii.header.get_zooms()))

    gt = torch.from_numpy(np.asarray(gt_nii.dataobj)).to(dtype=torch.long)
    prediction = torch.from_numpy(np.asarray(pred_nii.dataobj)).to(dtype=torch.long)
    gt_onehot = F.one_hot(gt.to(device), num_classes=k).movedim(-1, 0).bool().contiguous()
    prediction_onehot = F.one_hot(prediction.to(device), num_classes=k).movedim(-1, 0).bool().contiguous()

    surf_metrics = surface_metrics(gt_onehot[1:], prediction_onehot[1:], element_size=element_size)
    pprint(surf_metrics)
    print('\n')

    bord_metrics = border_metrics(gt_onehot[1:], prediction_onehot[1:], element_size=element_size)
    pprint(bord_metrics)
    print('\n')

    monai_hausdorff = compute_hausdorff_distance(prediction_onehot.unsqueeze(0), gt_onehot.unsqueeze(0),
                                                 spacing=element_size)
    print(f'Monai Hausdorff: {monai_hausdorff}')

    monai_hausdorff95 = compute_hausdorff_distance(prediction_onehot.unsqueeze(0), gt_onehot.unsqueeze(0),
                                                   percentile=95, directed=True, spacing=element_size)
    print(f'Monai Hausdorff95 (2 to 1): {monai_hausdorff95}')

    monai_asd = compute_average_surface_distance(prediction_onehot.unsqueeze(0), gt_onehot.unsqueeze(0),
                                                 spacing=element_size)
    print(f'Monai ASD (2 to 1): {monai_asd}')

    monai_assd = compute_average_surface_distance(prediction_onehot.unsqueeze(0), gt_onehot.unsqueeze(0),
                                                  symmetric=True, spacing=element_size)
    print(f'Monai ASSD: {monai_assd}')
