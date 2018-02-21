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
