import pytest
import torch

from distorch import distance_transform

images_sqdistances = (
    ([[0] * 5] * 5, [[float('inf')] * 5] * 5),  # by convention, infinite distance for empty image
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

devices = ['cpu']
if torch.cuda.is_available():
    devices.append('cuda')


@pytest.mark.parametrize('device_type', devices)
@pytest.mark.parametrize('use_pykeops', (False, True))
@pytest.mark.parametrize('image,sqdistances', images_sqdistances)
def test_euclidean_distance_transform(image, sqdistances, device_type: str, use_pykeops: bool):
    distance_transform.use_pykeops = use_pykeops
    device = torch.device(device_type)
    image = torch.tensor(image, dtype=torch.bool, device=device)
    distances = torch.tensor(sqdistances, dtype=torch.float, device=device).sqrt()
    edt = distance_transform.euclidean_distance_transform(image)
    assert torch.allclose(distances, edt), torch.stack((distances, edt, distances - edt), dim=0)
