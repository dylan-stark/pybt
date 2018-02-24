from copy import copy

import pytest

import numpy as np

from pybt.member import Member
from pybt.model import ModelWrapper

from test_model import KerasModel

class TestMember(object):
    def test_missing_args(self):
        with pytest.raises(TypeError):
            Member()
        with pytest.raises(TypeError):
            Member(model=KerasModel())
        with pytest.raises(TypeError):
            Member(model=KerasModel(), step_args={})
        with pytest.raises(TypeError):
            Member(model=KerasModel(), eval_args={})
        with pytest.raises(TypeError):
            Member(step_args={}, eval_args={})
        with pytest.raises(TypeError):
            Member(step_args={})
        with pytest.raises(TypeError):
            Member(eval_args={})

def test_step_history():
    images, labels = range(10), range(10)

    m = Member(ModelWrapper(KerasModel(), 1),
            step_args={'x': images, 'y': labels},
            eval_args={'x': images, 'y': labels})

    assert len(m._observations) == 0
    m.step()
    assert len(m._observations) == 1
    assert len(m._observations[0]) == 2 # epochs per step
    m.step()
    assert len(m._observations) == 2
    assert len(m._observations[1]) == 2 # epochs per step

def test_copy_history():
    x_train, y_train = range(10), range(10)
    x_val, y_val = range(10), range(10)

    m = Member(ModelWrapper(KerasModel(), 1),
        step_args={'x': x_train, 'y': y_train},
        eval_args={'x': x_val, 'y': y_val})

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

