import pytest

from pybt import Population

class TestPopulation(object):
    def test_no_models(self):
        with pytest.raises(NotImplementedError):
            pop = Population()

    def test_one_model(self):
        x = 42
        pop = Population(x)

        assert len(pop) == 1

    def test_two_models(self, simple_mnist_model):
        ms = [simple_mnist_model, simple_mnist_model]
        pop = Population(ms)

        assert len(pop) == 2

class TestTrain(object):
    def test_no_steps(self, simple_mnist_model):
        with pytest.raises(TypeError):
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

