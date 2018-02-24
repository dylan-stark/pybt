import pytest

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

