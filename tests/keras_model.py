from math import floor, sin

import numpy as np

class KerasModel:
    """A minimal Keras model stub."""

    class History:
        def __init__(self, history):
            self.history = history

    def __init__(self):
        self.metrics_names = ['acc', 'loss', 'val_acc', 'val_loss']
        self._history = {k: [0.] for k in self.metrics_names}

    def fit(self, x=None, y=None, epochs=None, initial_epoch=None,
            validation_data=None, batch_size=None):
        self._history = {
            'acc': np.array(
                [floor(sin(e)*100) for e in range(initial_epoch, epochs)]),
            'loss': np.array([e + 0.5 for e in range(initial_epoch, epochs)]),
            'val_acc':
                np.array([e + 0.25 for e in range(initial_epoch, epochs)]),
            'val_loss':
                np.array([e + 0.75 for e in range(initial_epoch, epochs)])
        }

        return self.History(self._history)

    def evaluate(self, x=None, y=None):
        return self._history['loss'][-1], self._history['acc'][-1]

