#!/usr/bin/env python3

# BSD 3-Clause License

# Copyright (c) 2025, Jérôme Rony, Hoel Kervadec

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import argparse
import os
from functools import partial
from pathlib import Path
from pprint import pprint

import nibabel as nib
import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from distorch.metrics import boundary_metrics


# # Assert utils
# def is_simplex(t: Tensor, axis=1) -> bool:
#     _sum = t.sum(axis).type(torch.float32)
#     return torch.allclose(_sum, _sum.new_ones(()))  # verify allclose(_sum, 1) with broadcasting


# def is_one_hot(t: Tensor, axis=1) -> bool:
#     return is_simplex(t, axis) and torch.isin(torch.unique(t), t.new_tensor([0, 1]), assume_unique=True).all().item()


# # Pre-processing utils
# def class2one_hot(seg: Tensor, K: int) -> Tensor:
#     assert torch.isin(u := torch.unique(seg), seg.new_tensor(list(range(K))), assume_unique=True).all(), (u, K)

#     b, *img_shape = seg.shape  # type: tuple[int, ...]
#     seg_one_hot = seg.new_zeros((b, K, *img_shape), dtype=torch.int32)
#     seg_one_hot.scatter_(1, seg[:, None, ...], 1)

#     assert seg_one_hot.shape == (b, K, *img_shape)
#     assert is_one_hot(seg_one_hot)

#     return seg_one_hot


# Others utils
class termcolors:
    RESET = '\033[39m\033[49m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    BRED = '\033[91m'
    BGREEN = '\033[92m'
    BYELLOW = '\033[93m'
    BBLUE = '\033[94m'
    BMAGENTA = '\033[95m'
    BCYAN = '\033[96m'


tc = termcolors


class TQDM(tqdm):
    @property
    def rate_(self) -> float:
        return self._ema_dn() / self._ema_dt() if self._ema_dt() else 0.


tqdm_ = partial(TQDM, dynamic_ncols=True,
                leave=False,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [' '{rate_fmt}{postfix}]')


class VolumeDataset(Dataset):
    def __init__(self, stems: list[str], ref_folder: Path, pred_folder: Path,
                 ref_extension: str, pred_extension: str, K: int,
                 /, quiet: bool = False) -> None:

        self.stems: list[str] = stems
        self.ref_folder: Path = ref_folder
        self.pred_folder: Path = pred_folder
        self.ref_extension: str = ref_extension
        self.pred_extension: str = pred_extension
        self.K: int = K

        if not quiet:
            print(f'{tc.BLUE}>>> Initializing Volume dataset with {len(self.stems)} volumes{tc.RESET}')
            print(f'>>  {self.ref_folder}, {self.pred_folder}')
            # pprint(self.stems)

        assert all((self.ref_folder / (s + self.ref_extension)).exists()
                   for s in self.stems), self.ref_folder
        assert all((self.pred_folder / (s + self.pred_extension)).exists()
                   for s in self.stems), self.pred_folder

        if not quiet:
            print(f'{tc.GREEN}> All stems found in both folders{tc.RESET}')

    def __len__(self):
        return len(self.stems)

    def __getitem__(self, index: int) -> dict[str, Tensor | tuple[float, ...] | str]:
        stem: str = self.stems[index]

        ref: np.ndarray
        pred: np.ndarray
        spacing: tuple[float, ...]
        match (self.ref_extension, self.pred_extension):
            case ('.png', '.png'):
                ref = np.asarray(Image.open(self.ref_folder / (stem + self.ref_extension)))
                pred = np.asarray(Image.open(self.pred_folder / (stem + self.pred_extension)))

                spacing = (1, 1)
            case ('.nii.gz', '.nii.gz') | ('.nii', '.nii.gz') | ('.nii.gz', '.nii'):
                ref_obj = nib.load(self.ref_folder / (stem + self.ref_extension))
                spacing = ref_obj.header.get_zooms()  # type: ignore
                ref = np.asarray(ref_obj.dataobj, dtype=int)  # type: ignore

                pred_obj = nib.load(self.pred_folder / (stem + self.pred_extension))
                assert spacing == pred_obj.header.get_zooms()  # type: ignore
                pred = np.asarray(pred_obj.dataobj, dtype=int)  # type: ignore
            case _:
                raise NotImplementedError(self.ref_extension, self.pred_extension)

        return {'ref': torch.as_tensor(ref, dtype=torch.int64),
                'pred': torch.as_tensor(pred, dtype=torch.int64),
                'voxelspacing': spacing,
                'stem': stem}


def compute_metrics(loader, metrics: dict[str, Tensor], device, K: int,
                    ignore: list[int] | None = None) -> dict[str, Tensor]:
    desc = '>> Computing'
    tq_iter = tqdm_(enumerate(loader), total=len(loader), desc=desc)
    with torch.no_grad():
        for j, data in tq_iter:
            ref: Tensor = data['ref'].to(device)
            pred: Tensor = data['pred'].to(device)
            voxelspacing: tuple[float, ...] = data['voxelspacing']

            assert pred.shape == ref.shape

            B, *scan_shape = ref.shape
            assert B == 1, (B, ref.shape)

            for k in range(K):
                if ignore is not None and k in ignore:
                    continue

                if set(metrics.keys()).intersection({'3d_hd', '3d_hd95', '3d_assd'}):
                    h = boundary_metrics((pred == k)[:, None, ...],
                                         (ref == k)[:, None, ...],
                                         weight_by_size=False,
                                         element_size=tuple(float(e) for e in voxelspacing))
                    if '3d_hd' in metrics.keys():
                        metrics['3d_hd'][j, k] = h.Hausdorff
                    if '3d_hd95' in metrics.keys():
                        metrics['3d_hd95'][j, k] = (h.Hausdorff95_1_to_2 + h.Hausdorff95_2_to_1) / 2
                    if '3d_assd' in metrics.keys():
                        metrics['3d_assd'][j, k] = h.AverageSymmetricSurfaceDistance

            tq_iter.set_postfix({'batch_shape': list(pred.shape),
                                 'voxelspacing': [f'{float(e):.3f}' for e in voxelspacing]})

    return metrics


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Compute metrics for a list of images')
    parser.add_argument('--ref_folder', type=Path, required=True)
    parser.add_argument('--pred_folder', type=Path, required=True)

    extension_choices = ['.nii.gz', '.png', '.npy', '.nii']
    parser.add_argument('--ref_extension', type=str, required=True, choices=extension_choices)
    parser.add_argument('--pred_extension', type=str, choices=extension_choices)

    parser.add_argument('--num_classes', '-K', '-C', type=int, required=True)
    parser.add_argument('--ignored_classes', type=int, nargs='*',
                        help="Classes to skip (for instance background, or any other non-predicted class).")
    parser.add_argument('--metrics', type=str, nargs='+', choices=['3d_hd', '3d_hd95', '3d_assd'],
                        help='The metrics to compute.')

    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing metrics output, without prompt.')

    parser.add_argument('--save_folder', type=Path, default=None, help='The folder where to save the metrics')

    args = parser.parse_args()

    if not args.pred_extension:
        args.pred_extension = args.ref_extension

    pprint(args.__dict__)

    return args


def main() -> None:
    args = get_args()

    device = torch.device('cuda') if torch.cuda.is_available() and not args.cpu else torch.device('cpu')
    K: int = args.num_classes

    # p.stem does not handle well .nii.gz:
    stems: list[str] = list(map(lambda p: str(p.name).removesuffix(args.ref_extension),
                                args.ref_folder.glob(f'*{args.ref_extension}')))

    total_volumes = len(stems)
    metrics: dict[str, Tensor] = {m: torch.zeros((total_volumes, K), dtype=torch.float32, device=device)
                                  for m in args.metrics}

    dt_set = VolumeDataset(stems, args.ref_folder, args.pred_folder,
                           args.ref_extension, args.pred_extension, args.num_classes,
                           quiet=False)
    loader = DataLoader(dt_set,
                        batch_size=1,
                        num_workers=len(os.sched_getaffinity(0)),
                        shuffle=False,
                        drop_last=False)

    metrics = compute_metrics(loader, metrics, device, K, ignore=args.ignored_classes)

    print(f'>>> {args.pred_folder}')
    for key, v in metrics.items():
        print(key, v.mean(dim=0).cpu().numpy())

    if args.save_folder:
        savedir: Path = Path(args.save_folder)
        savedir.mkdir(parents=True, exist_ok=True)
        for key, e in metrics.items():
            dest: Path = savedir / f'{key}.npy'
            assert not dest.exists() or args.overwrite

            np.save(dest, e.cpu().numpy())


if __name__ == '__main__':
    main()
