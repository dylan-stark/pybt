import pytest

import numpy as np
import pandas as pd

from pybt.model import ModelWrapper

class KerasModel:
    """A minimal Keras model stub."""

    class History:
        def __init__(self, history):
            self.history = history

    def __init__(self):
        self.metrics_names = ['acc', 'val']

    def fit(self, x=None, y=None, epochs=None, initial_epoch=None):
        history = {
            'acc': np.array([e for e in range(initial_epoch, epochs)]),
            'val': np.array([e + 0.5 for e in range(initial_epoch, epochs)])
        }

        return self.History(history)

    def evaluate(self, x=None, y=None):
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

def test_as_data_frame():
    good_result = pd.DataFrame({
        'epoch': [i for i in range(10)],
        'acc': [i for i in range(10)],
        'val': [i+0.5 for i in range(10)]
    })

    m = ModelWrapper(KerasModel(), 1)
    h = KerasModel().fit(initial_epoch=0, epochs=10)
    df = m._history_as_data_frame(h.history, 0, 10)

    assert df.equals(good_result)

