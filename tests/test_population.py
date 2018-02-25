import numpy as np

import pytest

from pybt import Population

from test_model import KerasModel

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

class TestPopulation(object):
    def test_no_models(self):
        with pytest.raises(TypeError):
            pop = Population()

    def test_len_one_model(self):
        pop = Population(KerasModel(),
            eval_args={'x': range(10), 'y': range(10)})

        assert len(pop) == 1

    def test_str(self):
        s = str(Population(KerasModel()))
        assert s[:11] == 'Population:'

class TestTrain(object):
    def test_no_steps(self):
        with pytest.raises(TypeError):
            pop = Population(KerasModel(),
                eval_args={'x': range(10), 'y': range(10)})
            pop.train()

    def test_zero_steps(self):
        with pytest.raises(ValueError):
            pop = Population(KerasModel(),
                eval_args={'x': range(10), 'y': range(10)})
            pop.train(num_steps=0)

    def test_neg_steps(self):
        with pytest.raises(ValueError):
            pop = Population(KerasModel(),
                eval_args={'x': range(10), 'y': range(10)})
            pop.train(num_steps=-1)

    def test_final_model(self):
        x_train, y_train = range(10), range(10)
        x_val, y_val = range(10), range(10)
        x_test, y_test = range(10), range(10)

        pop = Population(KerasModel(),
            step_args={'x': x_train, 'y': y_train},
            eval_args={'x': x_val, 'y': y_val})
        m = pop.train(num_steps=1)
        loss, acc = m.evaluate(x_test, y_test)

class TestTidy(object):
    def test_no_train(self):
        pop = Population(KerasModel())
        assert len(pop.observations()) == 1

    def test_one_step(self):
        good_results = [
            {'member': 'm0', 'observations': []},
            {'member': 'm1', 'observations': [
                {'epochs': [0, 1],
                 'acc': [0, 1],
                 'loss': [0.5, 1.5],
                 'val_acc': [0.25, 1.25],
                 'val_loss': [0.75, 1.75],
                 't': 0}]}
        ]

        pop = Population(KerasModel())
        pop.train(num_steps=1)

        obs = pop.observations()

        print('result:\n{}'.format(obs))
        print('should be:\n{}'.format(good_results))

        compare_lists(obs, good_results)

    def test_two_steps(self):
        good_results = [
            {'member': 'm0', 'observations': []},
            {'member': 'm1', 'observations': [
                {'epochs': [0, 1],
                 'acc': [0, 1],
                 'loss': [0.5, 1.5],
                 'val_acc': [0.25, 1.25],
                 'val_loss': [0.75, 1.75],
                 't': 0}]},
            {'member': 'm2', 'observations': [
                {'epochs': [0, 1],
                 'acc': [0, 1],
                 'loss': [0.5, 1.5],
                 'val_acc': [0.25, 1.25],
                 'val_loss': [0.75, 1.75],
                 't': 0},
                {'epochs': [2, 3],
                 'acc': [2, 3],
                 'loss': [2.5, 3.5],
                 'val_acc': [2.25, 3.25],
                 'val_loss': [2.75, 3.75],
                 't': 1}]}
        ]

        pop = Population(KerasModel())
        pop.train(num_steps=2)

        obs = pop.observations()

        print('result:\n{}'.format(obs))
        print('should be:\n{}'.format(good_results))

        compare_lists(obs, good_results)

