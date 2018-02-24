import pytest

from pybt import Population

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

