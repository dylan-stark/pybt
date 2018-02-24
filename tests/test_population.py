import numpy as np

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
        results = np.array([[0.99, None,  None,  None]], dtype='float64')

        pop = Population(KerasModel())
        assert np.allclose(pop.asarray(), results, equal_nan=True)

    def test_one_step(self):
        results = np.array(
            [[0.99, None,  None,  None],
             [0.99, 0.  , 0.  , 0.5 ],
             [0.99, 1.  , 1.  , 1.5 ]],
            dtype='float64')

        pop = Population(KerasModel())
        pop.train(num_steps=1)
        assert np.allclose(pop.asarray(), results, equal_nan=True)

