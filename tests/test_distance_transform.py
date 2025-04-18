import pytest
import torch

pytest.importorskip('cupy')
pytest.importorskip('cucim')
from distorch import distance_transform

# @formatter:off
images_sqdistances = (
    ([[0] * 5] * 5, [[float('inf')] * 5] * 5),  # by convention, infinite distance for empty image
    ([[1] * 5] * 5, [[0] * 5] * 5),
    ([[0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0],
      [0, 0, 1, 0, 0],
      [0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0]],
     [[8, 5, 4, 5, 8],
      [5, 2, 1, 2, 5],
      [4, 1, 0, 1, 4],
      [5, 2, 1, 2, 5],
      [8, 5, 4, 5, 8]]),
    ([[0, 0, 1, 0, 0, 0],
      [0, 0, 0, 1, 0, 0],
      [0, 0, 1, 1, 0, 0],
      [0, 1, 0, 1, 0, 0],
      [0, 0, 0, 0, 0, 0]],
     [[4, 1, 0, 1, 2, 5],
      [5, 2, 1, 0, 1, 4],
      [2, 1, 0, 0, 1, 4],
      [1, 0, 1, 0, 1, 4],
      [2, 1, 2, 1, 2, 5]]),
    ([[[0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0],
       [0, 0, 0, 1, 0],
       [0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0]],
      [[0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0],
       [0, 1, 0, 0, 0],
       [0, 0, 0, 0, 0]]],
     [[[13, 8, 5, 4, 5],
       [10, 5, 2, 1, 2],
       [ 9, 4, 1, 0, 1],
       [10, 5, 2, 1, 2],
       [13, 8, 5, 4, 5]],
      [[10, 9, 10, 13, 18],
       [ 5, 4,  5,  8, 13],
       [ 2, 1,  2,  5, 10],
       [ 1, 0,  1,  4,  9],
       [ 2, 1,  2,  5, 10]]]
     )
)
# @formatter:on


@pytest.mark.parametrize('image,sqdistances', images_sqdistances)
def test_euclidean_distance_transform(image, sqdistances, device_type: str = 'cuda'):
    device = torch.device(device_type)
    image = torch.tensor(image, dtype=torch.bool, device=device)
    distances = torch.tensor(sqdistances, dtype=torch.float, device=device).sqrt()
    edt = distance_transform.euclidean_distance_transform(image)
    assert torch.allclose(distances, edt), torch.stack((distances, edt, distances - edt), dim=0)


# @formatter:off
surface_images_sqdistances = (
    ([[0] * 5] * 5, [[float('inf')] * 6] * 6),  # by convention, infinite distance for empty image
    ([[1] * 5] * 5,  # full image -> image border is considered surface
     [[0, 0, 0, 0, 0, 0],
      [0, 1, 1, 1, 1, 0],
      [0, 1, 4, 4, 1, 0],
      [0, 1, 4, 4, 1, 0],
      [0, 1, 1, 1, 1, 0],
      [0, 0, 0, 0, 0, 0]]),
    ([[0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0],
      [0, 0, 1, 0, 0],
      [0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0]],
     [[8, 5, 4, 4, 5, 8],
      [5, 2, 1, 1, 2, 5],
      [4, 1, 0, 0, 1, 4],
      [4, 1, 0, 0, 1, 4],
      [5, 2, 1, 1, 2, 5],
      [8, 5, 4, 4, 5, 8]]),
    ([[0, 0, 1, 0, 0, 0],
      [0, 0, 0, 1, 0, 0],
      [0, 0, 1, 1, 0, 0],
      [0, 1, 0, 1, 0, 0],
      [0, 0, 0, 0, 0, 0]],
     [[4, 1, 0, 0, 1, 2, 5],
      [4, 1, 0, 0, 0, 1, 4],
      [2, 1, 0, 0, 0, 1, 4],
      [1, 0, 0, 0, 0, 1, 4],
      [1, 0, 0, 0, 0, 1, 4],
      [2, 1, 1, 1, 1, 2, 5]]),
    ([[[0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0],
       [0, 0, 0, 1, 0],
       [0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0]],
      [[0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0],
       [0, 1, 0, 0, 0],
       [0, 0, 0, 0, 0]]],
     [[[13, 8, 5, 4, 4, 5],
       [10, 5, 2, 1, 1, 2],
       [ 9, 4, 1, 0, 0, 1],
       [ 9, 4, 1, 0, 0, 1],
       [10, 5, 2, 1, 1, 2],
       [13, 8, 5, 4, 4, 5]],
      [[10, 9, 9, 10, 13, 18],
       [ 5, 4, 4,  5,  8, 13],
       [ 2, 1, 1,  2,  5, 10],
       [ 1, 0, 0,  1,  4,  9],
       [ 1, 0, 0,  1,  4,  9],
       [ 2, 1, 1,  2,  5, 10]]]
     )
)
# @formatter:on

@pytest.mark.parametrize('image,sqdistances', surface_images_sqdistances)
def test_surface_euclidean_distance_transform(image, sqdistances, device_type: str = 'cuda'):
    device = torch.device(device_type)
    image = torch.tensor(image, dtype=torch.bool, device=device)
    distances = torch.tensor(sqdistances, dtype=torch.float, device=device).sqrt()
    edt = distance_transform.surface_euclidean_distance_transform(image)
    assert torch.allclose(distances, edt), torch.stack((distances, edt, distances - edt), dim=0)
