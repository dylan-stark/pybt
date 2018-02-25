import numpy as np
import pandas as pd

import pytest

from pybt import Population

from test_model import KerasModel

class TestPopulation(object):
    def test_no_models(self):
        with pytest.raises(NotImplementedError):
            pop = Population()

    def test_one_model(self):
        pop = Population(KerasModel(),
            eval_args={'x': range(10), 'y': range(10)})

        assert len(pop) == 1

    def test_two_models(self):
        pop = Population([KerasModel(), KerasModel()],
            eval_args={'x': range(10), 'y': range(10)})

        assert len(pop) == 2

class TestTrain(object):
    def test_no_steps(self):
        with pytest.raises(TypeError):
            pop = Population([KerasModel(), KerasModel()],
                eval_args={'x': range(10), 'y': range(10)})
            pop.train()

    def test_zero_steps(self):
        with pytest.raises(ValueError):
            pop = Population([KerasModel(), KerasModel()],
                eval_args={'x': range(10), 'y': range(10)})
            pop.train(num_steps=0)

    def test_neg_steps(self):
        with pytest.raises(ValueError):
            pop = Population([KerasModel(), KerasModel()],
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
        assert pop.as_data_frame().empty

    def test_one_step(self):
        good_results = pd.DataFrame({
            'model': ['m1', 'm1'],
            't': [0, 0],
            'epoch': [0, 1],
            'acc': [0.0, 1.0],
            'loss': [0.5, 1.5]
        })

        pop = Population(KerasModel())
        pop.train(num_steps=1)

        df = pop.as_data_frame()
        df = df.reindex_axis(sorted(df.columns), axis=1)
        print('result:\n{}'.format(df))
        print('should be:\n{}'.format(good_results))
        assert df.equals(good_results)

    def test_two_steps(self):
        good_results = pd.DataFrame({
            'model': ['m1', 'm1', 'm2', 'm2', 'm2', 'm2'],
            't': [0, 0, 0, 0, 1, 1],
            'epoch': [0, 1, 0, 1, 2, 3],
            'acc': [0.0, 1.0, 0.0, 1.0, 2.0, 3.0],
            'loss': [0.5, 1.5, 0.5, 1.5, 2.5, 3.5]
        })

        pop = Population(KerasModel())
        pop.train(num_steps=2)
        df = pop.as_data_frame()
        df = df.reindex_axis(sorted(df.columns), axis=1)
        print('result:\n{}'.format(df))
        print('should be:\n{}'.format(good_results))
        assert df.equals(good_results)

