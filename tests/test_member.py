import logging

from copy import copy

import pytest

import numpy as np

from pybt.member import Member

from conftest import SimpleWrapper

def test_missing_args(keras_model):
    with pytest.raises(TypeError):
        Member()
    with pytest.raises(TypeError):
        Member(model=keras_model)
    with pytest.raises(TypeError):
        Member(model=keras_model, step_args={})
    with pytest.raises(TypeError):
        Member(model=keras_model, eval_args={})
    with pytest.raises(TypeError):
        Member(step_args={}, eval_args={})
    with pytest.raises(TypeError):
        Member(step_args={})
    with pytest.raises(TypeError):
        Member(eval_args={})

def test_step_history(keras_model):
    images, labels = range(10), range(10)

    m = Member(SimpleWrapper(keras_model), 1,
            step_args={'epochs_per_step': 2,
                'fit_args': {'x': images, 'y': labels}},
            eval_args={'x': images, 'y': labels})

    assert len(m.observations()['observations']) == 0
    m.step()
    assert len(m.observations()['observations']) == 1
    m.step()
    assert len(m.observations()['observations']) == 2

def test_copy_history(keras_model):
    x_train, y_train = range(10), range(10)
    x_val, y_val = range(10), range(10)

    m = Member(SimpleWrapper(keras_model), 1,
        step_args={'epochs_per_step': 2,
            'fit_args': {'x': x_train, 'y': y_train}},
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

def test_member_done(keras_model):
    from pybt.policy.done import StopAfter

    m = Member(SimpleWrapper(keras_model), 1,
        step_args={'epochs_per_step': 2, 'fit_args': {'x': {}, 'y': {}}},
        eval_args={}, stopping_criteria=StopAfter(epochs=4))

    assert m.done() == False
    m.step()
    assert m.done() == False
    m.step()
    assert m.done() == True

def test_member_ready(keras_model):
    """This member should be ready after the sixth and twelth epochs."""
    from pybt.policy.ready import ReadyAfter
    from pybt.policy.done import StopAfter

    m = Member(SimpleWrapper(keras_model), 1,
        stopping_criteria=StopAfter(epochs=13),
        ready_strategy=ReadyAfter(epochs=6),
        step_args={'epochs_per_step': 3, 'fit_args': {'x': {}, 'y': {}}},
        eval_args={})

    assert m.ready() == False
    m.step()
    assert m.ready() == False
    m.step()
    assert m.ready() == True
    m.step()
    assert m.ready() == False
    m.step()
    assert m.ready() == True
    m.step()
    assert m.ready() == False

