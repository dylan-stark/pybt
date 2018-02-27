from copy import copy

import pytest

import numpy as np
import pandas as pd

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

    m = Member(ModelWrapper(KerasModel()), 1,
            step_args={'x': images, 'y': labels},
            eval_args={'x': images, 'y': labels})

    assert len(m.observations()['observations']) == 0
    m.step()
    assert len(m.observations()['observations']) == 1
    m.step()
    assert len(m.observations()['observations']) == 2

def test_copy_history():
    x_train, y_train = range(10), range(10)
    x_val, y_val = range(10), range(10)

    m = Member(ModelWrapper(KerasModel()), 1,
        step_args={'x': x_train, 'y': y_train},
        eval_args={'x': x_val, 'y': y_val})

    m.step()
    m.step()
    assert len(m.observations()['observations']) == 2

    m2 = copy(m)
    assert len(m2.observations()['observations']) == 2

    m2.step()
    m2.step()
    assert len(m.observations()['observations']) == 2
    assert len(m2.observations()['observations']) == 4

def test_member_done():
    from pybt.member import StopAfter

    m = Member(ModelWrapper(KerasModel()), 1, step_args={}, eval_args={},
        stopping_criteria=StopAfter(epochs=4))

    assert m.done() == False
    m.step()
    assert m.done() == False
    m.step()
    assert m.done() == True

