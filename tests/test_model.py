import torch

from libs.model import VGG11


def test_vgg11_io_shape():
    model = VGG11()
    x = torch.rand(24, 3, 224, 224)
    y = model(x)
    assert y.shape == (24, 12)
