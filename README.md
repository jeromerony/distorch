# DisTorch: A fast GPU implementation of 3D Hausdorff Distance

**Disclaimer: this library is in a pre-alpha state. Dependencies, features and API are expected to change.
Additionally, current development is done on the `main` branch.
This will change once we reach a stable version and produce a release on PyPI.**

## Installation

This library relies on CUDA, through `Triton` and/or `KeOps` packages.
It is expected that `PyTorch` is installed with GPU support, which can be verified with
`python -c "import torch; print(torch.cuda.is_available())"`.
We only provide minimal CPU support, mainly for debugging purposes.

To install the library with `pip`:
```bash
pip install git+https://github.com/jeromerony/distorch.git
```
or clone the repository, `cd` into it and run `pip install .`.

If you want to install the library in editable mode, with IDE indexing compatibility:
```bash
pip install -e . --config-settings editable_mode=compat
```

### KeOps compatibility

You might run into compilation issues involving `KeOps`, [which requires a C++ compiler](https://www.kernel-operations.io/keops/python/installation.html#compilation-issues).
If you are using `anaconda` for your environment, you can solve this by installing `g++` in your environment (for `KeOps=2.3`):
```bash
conda install 'conda-forge::gxx>=14.2.0'
```

## Overview

This repository contains the code library presented in our MIDL 2025 submission to the short paper track.
It implements the Hausdorff distance, and similar distance based metrics, with GPU accelerated frameworks (currently [`KeOps`](https://www.kernel-operations.io/) and [`Triton`](https://github.com/triton-lang/triton)).

This library is destined to researchers who want to evaluate the quality of segmentations (2D or 3D) w.r.t. a ground truth, according to distance metrics such as the Hausdorff distance, the Average Symmetric Surface Distance (ASSD), etc.
In particular, doing this evaluation for 3D volumes can be challenging in terms of computation time, requiring several seconds per volume with CPU implementations.

Here, we provide an implementation of these metrics that leverages CUDA, managing to be faster or on-par with other libraries.
The goal of our implementation is 3-fold:
- be fast on GPU for 3D volumes
- be easy to install, minimizing the amount of dependencies
- be easy to inspect
Additional care is taken to provide accurate results, although the ASSD metric is currently not evaluated correctly by any library, including ours. More details in [CORRECTNESS.md](CORRECTNESS.md).

Our implementation is particularly fast on GPU, especially for small objects, such as the WMH 1.0 dataset.
Citing Table 1 from our submission, on three datasets (SegTHOR, OAI, WMH 1.0):

|                   | Runtime (ms) | Mem. (GiB) | Runtime (ms) | Mem. (GiB) | Runtime (ms) | Mem. (GiB) |
|-------------------|-------------:|-----------:|-------------:|-----------:|-------------:|-----------:|
| MedPy             |    2.6 × 10⁴ |         NA |    1.8 × 10⁴ |         NA |          296 |         NA |
| MeshMetrics       |    8.5 × 10³ |         NA |    1.2 × 10⁴ |         NA |          436 |         NA |
| Monai             |          723 |        4.7 |    1.7 × 10³ |        2.1 |         52.2 |       0.52 |
| Monai w/ cuCIM    |         24.9 |        2.6 |         22.4 |       0.95 |          6.3 |       0.09 |
| DisTorch (Keops)  |         29.1 |        1.7 |         26.8 |       0.62 |          1.4 |       0.06 |
| DisTorch (Triton) |         29.5 |        1.7 |         35.4 |       0.62 |          1.4 |       0.06 |


## Usage

The core functions for metrics computation in the [`metrics.py`](distorch/metrics.py) file, but we also provide some utility to compute the desired metrics between two folders:
```bash
>>> python compute_metrics.py --help
usage: compute_metrics.py [-h] --ref_folder REF_FOLDER --pred_folder PRED_FOLDER --ref_extension {.nii.gz,.png,.npy,.nii} [--pred_extension {.nii.gz,.png,.npy,.nii}]
                          --num_classes NUM_CLASSES [--metrics {3d_hd,3d_hd95,3d_assd} [{3d_hd,3d_hd95,3d_assd} ...]] [--cpu] [--overwrite] [--save_folder SAVE_FOLDER]

Compute metrics for a list of images

options:
  -h, --help            show this help message and exit
  --ref_folder REF_FOLDER
  --pred_folder PRED_FOLDER
  --ref_extension {.nii.gz,.png,.npy,.nii}
  --pred_extension {.nii.gz,.png,.npy,.nii}
  --num_classes, -K, -C NUM_CLASSES
  --metrics {3d_hd,3d_hd95,3d_assd} [{3d_hd,3d_hd95,3d_assd} ...]
                        The metrics to compute.
  --cpu
  --overwrite           Overwrite existing metrics output, without prompt.
  --save_folder SAVE_FOLDER
                        The folder where to save the metrics
```

With an example invocation:
```bash
>>>  CUDA_VISIBLE_DEVICES=0 python -O compute_metrics.py --ref_folder ~/code/constrained_cnn/data/OAI/test/gt_3d --pred_folder ~/code/constrained_cnn/results/OAI/cross_entropy/best_epoch/test_3d/ --ref_extension .nii.gz -K 5 --metrics 3d_hd 3d_hd95 3d_assd

{'cpu': False,
 'metrics': ['3d_hd', '3d_hd95', '3d_assd'],
 'num_classes': 5,
 'overwrite': False,
 'pred_extension': '.nii.gz',
 'pred_folder': PosixPath('/home/hoel/code/constrained_cnn/results/OAI/cross_entropy/best_epoch/test_3d'),
 'ref_extension': '.nii.gz',
 'ref_folder': PosixPath('/home/hoel/code/constrained_cnn/data/OAI/test/gt_3d'),
 'save_folder': None}
>>> Initializing Volume dataset with 100 volumes
>>  /home/hoel/code/constrained_cnn/data/OAI/test/gt_3d, /home/hoel/code/constrained_cnn/results/OAI/cross_entropy/best_epoch/test_3d
> All stems found in both folders
>>> /home/hoel/code/constrained_cnn/results/OAI/cross_entropy/best_epoch/test_3d
3d_hd [7.522784  4.030704  5.530623  4.173739  5.6819477]
3d_hd95 [0.54261297 0.8277547  0.9859997  0.8178589  1.3280611 ]
3d_assd [0.07726964 0.25085682 0.31391588 0.25151908 0.35778362]
```


## License and citation
This repository is under the [BSD 3](LICENSE) license. For citation, currently the following may be used in LaTeX documents:
```bibtex
@misc{distorch,
  author = {Jérôme Rony and Hoel Kervadec},
  title = {DisTorch: A fast GPU implementation of 3D Hausdorff Distance},
  note = {Accessed on 2025-MM-DD},
  year = {2025},
  url = {https://github.com/jeromerony/distorch}
}
