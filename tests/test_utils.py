import pytest
import torch
import torch.nn.functional as F

from distorch.utils import zero_padded_nonnegative_quantile


@pytest.mark.parametrize('n', [17, 51, 100, 2048])
@pytest.mark.parametrize('sparsity', [0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95])
@pytest.mark.parametrize('q', [0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95])
def test_zero_padded_nonnegative_quantile(n: int, sparsity: float, q: float):
    k = int(n * (1 - sparsity))
    x = torch.randn(k).abs_()
    x_padded = F.pad(x, (n - k, 0), value=0)

    quantile_padded = torch.quantile(x_padded, q=q)
    quantile_adjusted = zero_padded_nonnegative_quantile(x, q=q, n=n)
    assert torch.allclose(quantile_padded, quantile_adjusted, atol=1e-5, rtol=0)
