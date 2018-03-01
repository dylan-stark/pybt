import pytest

from math import floor, sin

import numpy as np
import pandas as pd

from conftest import compare_dicts
from conftest import LinearModel
from conftest import QuadraticModel

from conftest import SimpleWrapper

def test_linear_state_based_metrics():
    m = LinearModel()
    w = SimpleWrapper(m)

    assert w != m
   
    # History should reflect the change in metrics over for a window
    # specified by the epoch range but continuing from where the model
    # state was as of the last call to fit().
    assert w.eval({}) == (0., 0.)

    h = w.fit({'initial_epoch': 0, 'epochs': 3})
    compare_dicts(h, {
        'epochs': [e for e in range(0, 3)],
        'acc': [1, 2, 3],
        'loss': [-1, -2, -3],
        'val_acc': [1.5, 2.5, 3.5],
        'val_loss': [-1.5, -2.5, -3.5]})

    # Run for a few more epochs
    h = w.fit({'initial_epoch': 3, 'epochs': 5})
    compare_dicts(h, {
        'epochs': [e for e in range(3, 5)],
        'acc': [4, 5],
        'loss': [-4, -5],
        'val_acc': [4.5, 5.5],
        'val_loss': [-4.5, -5.5]})

    # And now run for a few more, but reset the epoch range -- the key here
    # is that the metrics should continue relative to the previous state,
    # not the epoch range
    h = w.fit({'initial_epoch': 1, 'epochs': 3})
    compare_dicts(h, {
        'epochs': [e for e in range(1, 4)],
        'acc': [6, 7, 8],
        'loss': [-6, -7, -8],
        'val_acc': [6.5, 7.5, 8.5],
        'val_loss': [-6.5, -7.5, -8.5]})

def test_linear_copy():
    from copy import copy

    m = LinearModel()
    w = SimpleWrapper(m)

    w_copy = copy(w)

    assert w != w_copy
    assert w._model != w_copy._model
    assert w.eval({}) == w_copy.eval({})

    # Move original model ahead 3 epochs
    h = w.fit({'initial_epoch': 0, 'epochs': 3})
    compare_dicts(h, {
        'epochs': [e for e in range(0, 3)],
        'acc': [1, 2, 3],
        'loss': [-1, -2, -3],
        'val_acc': [1.5, 2.5, 3.5],
        'val_loss': [-1.5, -2.5, -3.5]})

    assert w.eval({}) == (-3, 3)
    assert w.eval({}) != w_copy.eval({})

    # Move copy ahead 2 epochs
    h = w_copy.fit({'initial_epoch': 1, 'epochs': 3})
    compare_dicts(h, {
        'epochs': [e for e in range(1, 3)],
        'acc': [1, 2],
        'loss': [-1, -2],
        'val_acc': [1.5, 2.5],
        'val_loss': [-1.5, -2.5]})

    assert w_copy.eval({}) == (-2, 2)
    assert w.eval({}) != w_copy.eval({})

    # Move original ahead 1 epochs
    h = w.fit({'initial_epoch': 3, 'epochs': 4})
    compare_dicts(h, {
        'epochs': [e for e in range(3, 4)],
        'acc': [4],
        'loss': [-4],
        'val_acc': [4.5],
        'val_loss': [-4.5]})

    assert w.eval({}) == (-4, 4)
    assert w.eval({}) != w_copy.eval({})

    # Move copy ahead 2 epochs
    h = w_copy.fit({'initial_epoch': 3, 'epochs': 5})
    compare_dicts(h, {
        'epochs': [e for e in range(3, 5)],
        'acc': [3, 4],
        'loss': [-3, -4],
        'val_acc': [3.5, 4.5],
        'val_loss': [-3.5, -4.5]})

    assert w_copy.eval({}) == (-4, 4)
    assert w.eval({}) == w_copy.eval({})

def test_quadratic_model():
    m = QuadraticModel()
    w = SimpleWrapper(m)

    h = w.fit({'initial_epoch': 0, 'epochs': 5})
    compare_dicts(h, {
        'epochs': [e for e in range(0, 3)],
        'acc': [4, 1, 0, 1, 4],
        'loss': [-4, -1, 0, -1, -4],
        'val_acc': [4.5, 1.5, 0.5, 1.5, 4.5],
        'val_loss': [-4.5, -1.5, -0.5, -1.5, -4.5]})

def test_fit(keras_model):
    good_results = {
        'acc': [-76, -96, -28],
        'val_loss': [4.75, 5.75, 6.75],
        'loss': [4.5, 5.5, 6.5],
        'val_acc': [4.25, 5.25, 6.25],
        'epochs': [4, 5, 6]
    }

    m = SimpleWrapper(keras_model)
    obs = m.fit(fit_args={'initial_epoch': 4, 'epochs': 7})

    print('result:\n{}'.format(obs))
    print('should be:\n{}'.format(good_results))

    compare_dicts(obs, good_results)

