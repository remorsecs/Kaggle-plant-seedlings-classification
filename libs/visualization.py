import matplotlib
from matplotlib import pyplot as plt

matplotlib.use('Qt5Agg')


class LossPresenter(object):
    def __init__(self):
        self._train_loss = {}
        self._val_loss = {}

    def update(self, epoch, train_loss=None, val_loss=None):
        if train_loss is None and val_loss is None:
            raise ValueError('Neither `train_loss` nor `val_loss` passed.')

        if train_loss is not None:
            self._train_loss[epoch] = train_loss

        if val_loss is not None:
            self._val_loss[epoch] = val_loss

    @property
    def epoch_train_loss(self):
        return list(self._train_loss.keys()), list(self._train_loss.values())

    @property
    def epoch_val_loss(self):
        return list(self._val_loss.keys()), list(self._val_loss.values())


class LossVisualizer(object):
    def __init__(self, presenter: LossPresenter = None):
        self._presenter = presenter

    def save(self, f):
        fig, ax = plt.subplots()
        ax.plot(*self._presenter.epoch_train_loss, label='Training')
        ax.plot(*self._presenter.epoch_val_loss, label='Validation')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        fig.savefig(f)
        plt.close(fig)
