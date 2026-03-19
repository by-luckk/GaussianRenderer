import torch

from gaussian_renderer import GaussianData, GaussianBatchData


def make_gaussian(n=10):
    return GaussianData(
        xyz=torch.randn(n, 3),
        rot=torch.randn(n, 4),
        scale=torch.randn(n, 3),
        opacity=torch.randn(n),
        sh=torch.randn(n, 16, 3),
    )


def test_len():
    assert len(make_gaussian(10)) == 10


def test_device():
    gd = make_gaussian()
    assert gd.device == gd.xyz.device


def test_batch_len():
    gd = GaussianBatchData(
        xyz=torch.randn(4, 10, 3),
        rot=torch.randn(4, 10, 4),
        scale=torch.randn(4, 10, 3),
        opacity=torch.randn(4, 10),
        sh=torch.randn(4, 10, 16, 3),
    )
    assert len(gd) == 10
    assert gd.batch_size == 4
    assert gd.device == gd.xyz.device
