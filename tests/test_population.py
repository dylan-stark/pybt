import pytest

from pybt import Population

class TestPopulation(object):
    def test_no_models(self):
        pop = Population()

        assert pop._models == None

    def test_one_model(self):
        x = 42
        pop = Population(x)

        assert len(pop._models) == 1
        assert pop._models[0] == x

    def test_two_models(self):
        ms = [42, 1234]
        pop = Population(ms)

        assert len(pop._models) == 2
        assert pop._models == ms

class TestTrain(object):
    def test_no_steps(self):
        with pytest.raises(TypeError):
            pop = Population()
            pop.train()

    def test_zero_steps(self):
        with pytest.raises(ValueError):
            pop = Population()
            pop.train(num_steps=0)

    def test_neg_steps(self):
        with pytest.raises(ValueError):
            pop = Population()
            pop.train(num_steps=-1)

