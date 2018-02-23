"""Generate sample of MNIST data for testing.

Usage:
    python tests/utils/mnist_sample.py
"""

import numpy as np

from keras.datasets import mnist
from keras.utils import to_categorical

num_samples = 10

(train_images, train_labels), _ = mnist.load_data()
train_images = train_images[:num_samples, :, :]
train_labels = train_labels[:num_samples]

train_images = train_images.reshape((num_samples, 28 * 28))
train_images = train_images.astype('float32') / 255

train_labels = to_categorical(train_labels)

np.save('tests/data/mnist_sample_images.npy', train_images)
np.save('tests/data/mnist_sample_labels.npy', train_labels)

