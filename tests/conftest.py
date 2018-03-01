import pytest

import numpy as np

from simple_wrapper import SimpleWrapper
from quadratic_model import QuadraticModel
from linear_model import LinearModel
from keras_model import KerasModel

def compare_lists(a, b):
    for e1, e2 in zip(a, b):
        if isinstance(e1, list) and isinstance(e2, list):
            compare_lists(e1, e2)
        elif isinstance(e1, dict) and isinstance(e2, dict):
            compare_dicts(e1, e2)
        else:
            assert e1 == e2

def compare_dicts(a, b):
    assert a.keys() == b.keys()
    for k in a.keys():
        e1, e2 = a[k], b[k]
        if isinstance(e1, list) and isinstance(e2, list):
            compare_lists(e1, e2)
        elif isinstance(e1, np.ndarray) or isinstance(e2, np.ndarray):
            assert np.allclose(e1, e2)
        elif isinstance(e1, dict) and isinstance(e2, dict):
            compare_dicts(e1, e2)
        else:
            assert e1 == e2

@pytest.fixture
def linear_model():
    return SimpleWrapper(LinearModel())

@pytest.fixture
def keras_model():
    return KerasModel()

@pytest.fixture
def keras_step_args():
    step_args = {
        'epochs_per_step': 2,
        'fit_args': {
            'x': list(range(10)),
            'y': list(range(10))
        }
    }

    return step_args

@pytest.fixture
def keras_eval_args():
    eval_args = {
        'x': list(range(10)),
        'y': list(range(10))
    }

    return eval_args

@pytest.fixture
def simple_mnist_model():
    from keras import models
    from keras import layers

    network = models.Sequential()
    network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
    network.add(layers.Dense(10, activation='softmax'))

    network.compile(optimizer='rmsprop',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    return network

@pytest.fixture
def mnist_sample():
    import numpy as np

    images = np.load('tests/data/mnist_sample_images.npy')
    labels = np.load('tests/data/mnist_sample_labels.npy')

    return (images, labels)

@pytest.fixture
def mnist_eval_args():
    x, y = mnist_sample()
    return {'x': x, 'y': y}

