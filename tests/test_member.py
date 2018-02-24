from copy import copy

import pytest

import numpy as np

from pybt.member import Member
from pybt.model import ModelWrapper

class TestMember(object):
    def test_missing_args(self, simple_mnist_model):
        with pytest.raises(TypeError):
            Member()
        with pytest.raises(TypeError):
            Member(model=simple_mnist_model)
        with pytest.raises(TypeError):
            Member(model=simple_mnist_model, step_args={})
        with pytest.raises(TypeError):
            Member(model=simple_mnist_model, eval_args={})
        with pytest.raises(TypeError):
            Member(step_args={}, eval_args={})
        with pytest.raises(TypeError):
            Member(step_args={})
        with pytest.raises(TypeError):
            Member(eval_args={})

def test_step_history(simple_mnist_model, mnist_sample):
    images, labels = mnist_sample
    m = Member(ModelWrapper(simple_mnist_model, 1),
            step_args={'x': images, 'y': labels},
            eval_args={'x': images, 'y': labels})

    assert len(m._observations) == 0
    m.step()
    assert len(m._observations) == 1
    assert len(m._observations[0]) == 2 # epochs per step
    m.step()
    assert len(m._observations) == 2
    assert len(m._observations[1]) == 2 # epochs per step

def test_copy_history(simple_mnist_model, mnist_sample):
    images, labels = mnist_sample
    m = Member(ModelWrapper(simple_mnist_model, 1),
            step_args={'x': images, 'y': labels},
            eval_args={'x': images, 'y': labels})

    m.step()
    m.step()
    assert len(m._observations) == 2
    assert len(m._observations[1]) == 2 # epochs per step

    m2 = copy(m)
    assert len(m2._observations) == 2
    assert len(m2._observations[1]) == 2 # epochs per step
    assert np.allclose(m._observations, m2._observations, equal_nan=True)
    m2.step()
    assert np.allclose(m._observations, m2._observations[:2], equal_nan=True)

