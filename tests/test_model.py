import pytest

import numpy as np

from pybt.model import ModelWrapper

class KerasModel:
    """A minimal Keras model stub."""

    class History:
        def __init__(self, history):
            self.history = history

    def __init__(self):
        self.metrics_names = ['acc', 'val']

    def fit(self, epochs, initial_epoch):
        history = {
            'acc': np.array([e for e in range(initial_epoch, epochs)]),
            'val': np.array([e + 0.5 for e in range(initial_epoch, epochs)])
        }

        return self.History(history)

    def evaluate(self):
        return 0.01, 0.99

def test_str():
    m = ModelWrapper(KerasModel(), 2)
    s = str(m)

    assert s == 'PyBT Model 2'

def test_fit():
    m = ModelWrapper(KerasModel(), 2)
    obs = m.fit(fit_args={'initial_epoch': 4, 'epochs': 7})

    assert np.allclose(obs,
        [[4., 4., 4.5], [5., 5., 5.5], [6., 6., 6.5]])

