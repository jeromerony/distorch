# DisTorch: A fast GPU implementation of 3D Hausdorff Distance
## (and others)

This repository contains the code of our MIDL 2025 submission to the short paper track. It implements the Hausdorff distance, and similar distance based metrics, with GPU accelerated frameworks (currently [`KeOps`](https://www.kernel-operations.io/) and [`Triton`](https://github.com/triton-lang/triton)).

Citing Table 1 from our submission, on three datasets (SegTHOR, OAI, WMH 1.0):

|                         |  Runtime (ms)               |  Mem. (GiB)  |  Runtime (ms)               |  Mem. (GiB) |  Runtime (ms) |  Mem. (GiB) |
|-------------------------|------------------------|--------|------------------------|-------|----------|-------|
| MedPy                   |  2.6 × 10⁴    |   NA    |  1.8 × 10⁴    |   NA   |  296   |  NA |
| MeshMetrics             |  8.5 × 10³    |   NA    |  1.2 × 10⁴    |   NA   |  436   |  NA |
| Monai                   |  723                 |  4.7   |  1.7 × 10³    |   2.1 |  52.2  |  0.52 |
| Monai w/ \texttt{cuCIM} |  24.9                |  2.6   |  22.4                |  0.95 |  6.3   |  0.09 |
| Ours                    |  29.1                |  1.7   |  26.8                |  0.62 |  1.4   |  0.06 |

## Installation
todo
## Usage
todo


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
