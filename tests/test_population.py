import numpy as np

import pytest

from pybt import Population

from test_model import KerasModel

class TestPopulation(object):
    def test_no_models(self):
        with pytest.raises(NotImplementedError):
            pop = Population()

    def test_one_model(self, simple_mnist_model, mnist_eval_args):
        pop = Population(simple_mnist_model, eval_args=mnist_eval_args)

        assert len(pop) == 1

    def test_two_models(self, simple_mnist_model, mnist_eval_args):
        ms = [simple_mnist_model, simple_mnist_model]
        pop = Population(ms, eval_args=mnist_eval_args)

        assert len(pop) == 2

class TestTrain(object):
    def test_no_steps(self, simple_mnist_model):
        with pytest.raises(ValueError):
            pop = Population(simple_mnist_model)
            pop.train()

    def test_zero_steps(self, simple_mnist_model):
        with pytest.raises(ValueError):
            pop = Population(simple_mnist_model)
            pop.train(num_steps=0)

    def test_neg_steps(self, simple_mnist_model):
        with pytest.raises(ValueError):
            pop = Population(simple_mnist_model)
            pop.train(num_steps=-1)

    def test_final_model(self, simple_mnist_model, mnist_sample):
        images, labels = mnist_sample

        pop = Population(simple_mnist_model,
            step_args={'x': images, 'y': labels},
            eval_args={'x': images, 'y': labels})
        m = pop.train(num_steps=1)
        loss, acc = m.evaluate(images, labels)

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

