from pathlib import Path

import torch
import torch.distributions as D
import matplotlib

from libs.visualization import LossPresenter, LossVisualizer

matplotlib.use('Qt5Agg')


def test_loss_presenter():
    epoch_expected = []

    m = D.Normal(0., 0.01)
    train_loss_expected = 2.5 - torch.log(torch.arange(1, 11, dtype=torch.float32)) + m.sample((10,))
    val_loss_expected = 3.0 - torch.log(torch.arange(1, 11, dtype=torch.float32)) + m.sample((10,))
    presenter = LossPresenter()
    num_epochs = 10

    for epoch in range(num_epochs):
        epoch_expected.append(epoch)
        presenter.update(train_loss=train_loss_expected[epoch].item(), epoch=epoch)
        presenter.update(val_loss=val_loss_expected[epoch].item(), epoch=epoch)

    epoch_expected = torch.tensor(epoch_expected)
    train_epoch, train_loss = presenter.epoch_train_loss
    train_epoch, train_loss = torch.tensor(train_epoch), torch.tensor(train_loss)
    assert torch.allclose(train_epoch, epoch_expected)
    assert torch.allclose(train_loss, train_loss_expected)

    val_epoch, val_loss = presenter.epoch_val_loss
    val_epoch, val_loss = torch.tensor(val_epoch), torch.tensor(val_loss)
    assert torch.allclose(val_epoch, epoch_expected)
    assert torch.allclose(val_loss, val_loss_expected)


def test_loss_visualizer_save():
    m = D.Normal(0., 0.01)
    train_loss_expected = 2.5 - torch.log(torch.arange(1, 11, dtype=torch.float32)) + m.sample((10,))
    val_loss_expected = 3.0 - torch.log(torch.arange(1, 11, dtype=torch.float32)) + m.sample((10,))
    presenter = LossPresenter()
    num_epochs = 10

    for epoch in range(num_epochs):
        presenter.update(train_loss=train_loss_expected[epoch].item(), epoch=epoch)
        presenter.update(val_loss=val_loss_expected[epoch].item(), epoch=epoch)

    visualizer = LossVisualizer(presenter=presenter)
    visualizer.save(f='loss.png')

    assert Path('loss.png').exists()
