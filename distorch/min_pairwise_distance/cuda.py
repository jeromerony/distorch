import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _minimum_sqdistances_2d(x1_ptr,
                            x2_ptr,
                            min_dist_ptr,
                            m,
                            BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    col_idx = tl.program_id(1)
    num_cols = tl.num_programs(1)

    x1_row_start = x1_ptr + row_idx * 2
    x1_x = tl.load(x1_row_start)
    x1_y = tl.load(x1_row_start + 1)

    targets_offset = tl.arange(0, BLOCK_SIZE) * 2
    col = col_idx * BLOCK_SIZE

    mask = col * 2 + targets_offset < m * 2
    x2_x = tl.load(x2_ptr + col * 2 + targets_offset, mask=mask, other=float('inf'))
    x2_y = tl.load(x2_ptr + col * 2 + targets_offset + 1, mask=mask, other=float('inf'))
    dx = x1_x - x2_x
    dy = x1_y - x2_y
    sqdist = dx * dx + dy * dy
    min_sqdist = tl.min(sqdist, axis=0)

    result_ptr = min_dist_ptr + row_idx * num_cols + col_idx
    tl.store(result_ptr, min_sqdist)


@triton.jit
def _minimum_sqdistances_3d(x1_ptr,
                            x2_ptr,
                            min_dist_ptr,
                            m,
                            BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    col_idx = tl.program_id(1)
    num_cols = tl.num_programs(1)

    x1_row_start = x1_ptr + row_idx * 3
    x1_x = tl.load(x1_row_start)
    x1_y = tl.load(x1_row_start + 1)
    x1_z = tl.load(x1_row_start + 2)

    targets_offset = tl.arange(0, BLOCK_SIZE) * 3
    col = col_idx * BLOCK_SIZE

    mask = col * 3 + targets_offset < m * 3
    x2_x = tl.load(x2_ptr + col * 3 + targets_offset, mask=mask, other=float('inf'))
    x2_y = tl.load(x2_ptr + col * 3 + targets_offset + 1, mask=mask, other=float('inf'))
    x2_z = tl.load(x2_ptr + col * 3 + targets_offset + 2, mask=mask, other=float('inf'))
    dx = x1_x - x2_x
    dy = x1_y - x2_y
    dz = x1_z - x2_z
    sqdist = dx * dx + dy * dy + dz * dz
    min_sqdist = tl.min(sqdist, axis=0)

    result_ptr = min_dist_ptr + row_idx * num_cols + col_idx
    tl.store(result_ptr, min_sqdist)


def min_sqdist(x1: Tensor, x2: Tensor, BLOCK_SIZE: int = 2048) -> Tensor:
    d = x1.size(1)
    assert d == x2.size(1)
    n, m = x1.size(0), x2.size(0)
    BLOCK_SIZE = min(BLOCK_SIZE, triton.next_power_of_2(m))
    grid_cols = triton.cdiv(m, BLOCK_SIZE)
    min_distances = x1.new_empty(size=(n, grid_cols))
    if d == 2:
        _minimum_sqdistances_2d[(n, grid_cols)](x1, x2, min_distances, m, BLOCK_SIZE=BLOCK_SIZE)
    if d == 3:
        _minimum_sqdistances_3d[(n, grid_cols)](x1, x2, min_distances, m, BLOCK_SIZE=BLOCK_SIZE)
    return min_distances.amin(dim=1)


if __name__ == "__main__":
    import torch

    torch.manual_seed(42)
    n, m, d = 1000, 4000, 3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t1 = torch.randn(n, d, device=device)
    t2 = torch.randn(m, d, device=device)

    # Naive implementation
    distances = torch.cdist(t1, t2, p=2).square()
    min_distances_naive = distances.amin(dim=1)

    # Compute minimum distances using our optimized kernel
    min_distances = min_sqdist(t1, t2)
    print('Validating against naive implementation...')

    # Check if results are close
    is_close = torch.allclose(min_distances, min_distances_naive, rtol=1e-5, atol=1e-5)
    print(f'Results match: {is_close}')
    if not is_close:
        max_diff = (min_distances - min_distances_naive).abs().max().item()
        print(f'Maximum difference: {max_diff}')
