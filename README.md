# DisTorch: A fast GPU implementation of 3D Hausdorff Distance

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

### KeOps optional dependency

`KeOps` is now an optional dependency. It provides marginally better performance compared to using the `Triton` backend.
However, we find that fixing dependency issues that might arise from installing `KeOps` does not justify the small performance gain.
It can be useful for developing, as writing `Triton` kernels can be cumbersome.

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
- be fast on GPU for 3D volumes;
- be easy to install, minimizing the amount of dependencies;
- be easy to inspect.

Additional care is taken to provide accurate results, although the ASSD metric is currently not evaluated correctly by any library, including ours. More details in [CORRECTNESS.md](CORRECTNESS.md).

Our implementation is particularly fast on GPU, especially for small objects, such as the WMH 1.0 dataset.
Below is a comparison on three datasets (SegTHOR, OAI, WMH 1.0) with runtime and GPU memory usage:

|                   | Runtime (ms) | Mem. (GiB) | Runtime (ms) | Mem. (GiB) | Runtime (ms) | Mem. (GiB) |
|-------------------|-------------:|-----------:|-------------:|-----------:|-------------:|-----------:|
| MedPy             |    2.6 × 10⁴ |         NA |    1.8 × 10⁴ |         NA |          296 |         NA |
| MeshMetrics       |    8.5 × 10³ |         NA |    1.2 × 10⁴ |         NA |          436 |         NA |
| Monai             |          723 |        4.7 |    1.7 × 10³ |        2.1 |         52.2 |       0.52 |
| Monai w/ cuCIM    |         24.9 |        2.6 |         22.4 |       0.95 |          6.3 |       0.09 |
| DisTorch (Keops)  |         28.0 |        1.7 |         27.3 |       0.62 |          1.8 |       0.05 |
| DisTorch (Triton) |         27.0 |        1.7 |         34.2 |       0.62 |          1.4 |       0.05 |


## Usage

The core function of this library is the `distorch.boundary_metrics` function located in the `distorch/metric.py` module.
This function computes several distance-based metrics such as Hausdorff, ASSD and NSSD. It can be used for 2D images and 3D volumes:
- for 2D images, this function expects 2D or 3D binary tensors, where the leading dimension is the batch dimension in the latter case; 
- for 3D volumes, this function expects $`n`$D binary tensors, with $`n\geq 4`$, where all leading dimensions are batch dimensions.

Example usage is as follows:
```python
import torch
import distorch
device = torch.device('cuda')
A = torch.tensor([[0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0],
                  [0, 0, 1, 1, 0],
                  [0, 1, 1, 1, 0],
                  [0, 0, 0, 0, 0]], dtype=torch.bool, device=device)
B = torch.tensor([[1, 1, 0, 0, 0],
                  [0, 1, 1, 0, 0],
                  [0, 1, 1, 1, 0],
                  [0, 1, 1, 0, 0],
                  [0, 0, 0, 0, 0]], dtype=torch.bool, device=device)
metrics = distorch.boundary_metrics(A, B)
print(metrics)
```
```text
DistanceMetrics(Hausdorff=tensor(1.4142, device='cuda:0'),
                Hausdorff95_1_to_2=tensor(1., device='cuda:0'),
                Hausdorff95_2_to_1=tensor(1.1036, device='cuda:0'),
                AverageSurfaceDistance_1_to_2=tensor(0.2857, device='cuda:0'),
                AverageSurfaceDistance_2_to_1=tensor(0.4009, device='cuda:0'),
                AverageSymmetricSurfaceDistance=tensor(0.3471, device='cuda:0'),
                NormalizedSurfaceDistance_1_to_2=tensor(1., device='cuda:0'),
                NormalizedSurfaceDistance_2_to_1=tensor(0.9375, device='cuda:0'),
                NormalizedSymmetricSurfaceDistance=tensor(0.9667, device='cuda:0'))
```

## License and citation
This repository is under the [BSD 3](LICENSE) license. For citation, currently the following may be used in LaTeX documents:
```bibtex
@misc{distorch,
  author = {Jérôme Rony and Hoel Kervadec},
  title = {DisTorch: A fast GPU implementation of 3D Hausdorff Distance},
  year = {2025},
  url = {https://github.com/jeromerony/distorch}
}
