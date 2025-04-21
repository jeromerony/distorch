import pytest
import torch

from distorch import boundary

devices = ['cpu']
if torch.cuda.is_available():
    devices.append('cuda')

# @formatter:off
images_isborder = (
    ([[0] * 5] * 5, [[0] * 5] * 5),
    ([[1] * 5] * 5, [[1, 1, 1, 1, 1],  # full image -> image border is considered to be border
                     [1, 0, 0, 0, 1],
                     [1, 0, 0, 0, 1],
                     [1, 0, 0, 0, 1],
                     [1, 1, 1, 1, 1]]),
    ([[0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0],
      [0, 0, 1, 0, 0],
      [0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0]]),
    ([[0, 0, 1, 0, 1, 0],
      [0, 1, 1, 1, 0, 0],
      [0, 1, 1, 1, 0, 0],
      [0, 1, 1, 1, 0, 0],
      [0, 1, 0, 0, 0, 0]], [[0, 0, 1, 0, 1, 0],
                            [0, 1, 0, 1, 0, 0],
                            [0, 1, 0, 1, 0, 0],
                            [0, 1, 1, 1, 0, 0],
                            [0, 1, 0, 0, 0, 0]]),
    ([[[0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0],
       [0, 0, 0, 1, 0],
       [0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0]],
      [[0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0],
       [0, 1, 0, 0, 0],
       [0, 0, 0, 0, 0]]], [[[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]],
                           [[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0]]]
     )
)
# @formatter:on

@pytest.mark.parametrize('device_type', devices)
@pytest.mark.parametrize('image,is_border', images_isborder)
def test_is_border_element(image, is_border, device_type: str):
    device = torch.device(device_type)
    image = torch.tensor(image, dtype=torch.bool, device=device)
    is_border = torch.tensor(is_border, dtype=torch.bool, device=device)
    distorch_is_border = boundary.is_border_element(image)
    assert torch.equal(is_border, distorch_is_border), (is_border, distorch_is_border)


# @formatter:off
images_issurface = (
    ([[0] * 5] * 5, [[0] * 6] * 6),
    ([[1] * 5] * 5, [[1, 1, 1, 1, 1, 1],  # full image -> image border is considered to be surface
                     [1, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0, 1],
                     [1, 1, 1, 1, 1, 1]]),
    ([[0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0],
      [0, 0, 1, 0, 0],
      [0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0],
                         [0, 0, 1, 1, 0, 0],
                         [0, 0, 1, 1, 0, 0],
                         [0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0]]),
    ([[0, 0, 1, 0, 1, 0],
      [0, 1, 1, 1, 0, 0],
      [0, 1, 1, 1, 0, 0],
      [0, 1, 1, 1, 0, 0],
      [0, 1, 0, 0, 0, 0]], [[0, 0, 1, 1, 1, 1, 0],
                            [0, 1, 1, 1, 1, 1, 0],
                            [0, 1, 0, 0, 1, 0, 0],
                            [0, 1, 0, 0, 1, 0, 0],
                            [0, 1, 1, 1, 1, 0, 0],
                            [0, 1, 1, 0, 0, 0, 0]]),
    ([[[0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0],
       [0, 0, 0, 1, 0],
       [0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0]],
      [[0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0],
       [0, 1, 0, 0, 0],
       [0, 0, 0, 0, 0]]],[[[0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 1, 1, 0],
                           [0, 0, 0, 1, 1, 0],
                           [0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0]],
                          [[0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0],
                           [0, 1, 1, 0, 0, 0],
                           [0, 1, 1, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0]]]
     )
)
# @formatter:on

@pytest.mark.parametrize('device_type', devices)
@pytest.mark.parametrize('image,is_surface', images_issurface)
def test_is_surface_vertex(image, is_surface, device_type: str):
    device = torch.device(device_type)
    image = torch.tensor(image, dtype=torch.bool, device=device)
    is_surface = torch.tensor(is_surface, dtype=torch.bool, device=device)
    distorch_is_surface = boundary.is_surface_vertex(image)
    assert torch.equal(is_surface, distorch_is_surface), (is_surface, distorch_is_surface)
