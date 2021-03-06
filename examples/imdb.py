"""Classifying movie reviews.

This example is based on an example from François Chollet's Deep Learning
with Python that was also made available in the
`3.5-classifying-movie-reviews.ipynb` notebook at

    https://github.com/fchollet/deep-learning-with-python-notebooks
"""

import logging
logging.basicConfig(level=logging.DEBUG)

from keras import layers, models
from keras.datasets import imdb
from keras.layers import Dense
from keras.losses import binary_crossentropy
from keras.optimizers import RMSprop
from keras.utils import to_categorical

import numpy as np

from pybt import Trainer
from pybt.model import KerasModelWrapper
from pybt.policy.done import StopAfter
from pybt.policy.ready import ReadyAfter

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))

    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.

    return results

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(
    num_words=10000)

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

model = models.Sequential()
model.add(Dense(16, activation='relu', input_shape=(10000,)))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=RMSprop(lr=0.001),
              loss=binary_crossentropy,
              metrics=['accuracy'])

# Create a population with this model and train
m = KerasModelWrapper(model, optimizer=RMSprop(lr=0.001),
    loss=binary_crossentropy, metrics=['accuracy'])
t = Trainer(model=m, stopping_criteria=StopAfter(4),
    ready_strategy=ReadyAfter(1),
    step_args = {'epochs_per_step': 1,
        'fit_args': {'x': partial_x_train, 'y': partial_y_train}},
    eval_args = {'x': x_val, 'y': y_val})
model, score = t.train()

print(t)

test_loss, test_acc = model.evaluate(x_test, y_test)

print('Test accuracy after training is {}'.format(test_acc, test_loss))

