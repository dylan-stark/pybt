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
        self.metrics_names = ['acc', 'loss', 'val_acc', 'val_loss']

    def fit(self, x=None, y=None, epochs=None, initial_epoch=None):
        history = {
            'acc': np.array([e for e in range(initial_epoch, epochs)]),
            'loss': np.array([e + 0.5 for e in range(initial_epoch, epochs)]),
            'val_acc':
                np.array([e + 0.25 for e in range(initial_epoch, epochs)]),
            'val_loss':
                np.array([e + 0.75 for e in range(initial_epoch, epochs)])
        }

        return self.History(history)

    def evaluate(self, x=None, y=None):
        return 0.01, 0.99

def test_str():
    m = ModelWrapper(KerasModel())
    s = str(m)

    assert s[:7] == 'Model ('

def test_fit():
    good_results = pd.DataFrame({
        'epoch': [4, 5, 6],
        'acc': [4, 5, 6],
        'val_acc': [4.25, 5.25, 6.25],
        'loss': [4.5, 5.5, 6.5],
        'val_loss': [4.75, 5.75, 6.75]
    })

    m = ModelWrapper(KerasModel())
    obs = m.fit(fit_args={'initial_epoch': 4, 'epochs': 7})

    obs = obs.reindex(sorted(obs.columns), axis=1)

    print('result:\n{}'.format(obs))
    print('should be:\n{}'.format(good_results))

    obs.equals(good_results)

def test_as_data_frame():
    good_result = pd.DataFrame({
        'epoch': [i for i in range(10)],
        'acc': [i for i in range(10)],
        'loss': [i+0.5 for i in range(10)],
        'val_acc': [i+0.25 for i in range(10)],
        'val_loss': [i+0.75 for i in range(10)]
    })

    m = ModelWrapper(KerasModel())
    h = KerasModel().fit(initial_epoch=0, epochs=10)
    df = m._history_as_data_frame(h.history, 0, 10)

    assert df.equals(good_result)

