from copy import copy

import pytest

from pybt.member import Member

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
    m = Member(simple_mnist_model,
            step_args={'x': images, 'y': labels},
            eval_args={'x': images, 'y': labels})

    assert len(m._histories) == 0
    m.step()
    assert len(m._histories) == 1
    assert len(m._histories[0]['loss']) == 2 # epochs per step
    m.step()
    assert len(m._histories) == 2
    assert len(m._histories[1]['loss']) == 2 # epochs per step

def test_copy_history(simple_mnist_model, mnist_sample):
    images, labels = mnist_sample
    m = Member(simple_mnist_model,
            step_args={'x': images, 'y': labels},
            eval_args={'x': images, 'y': labels})

    m.step()
    m.step()
    assert len(m._histories) == 2
    assert len(m._histories[1]['loss']) == 2 # epochs per step

    m2 = copy(m)
    assert len(m2._histories) == 2
    assert len(m2._histories[1]['loss']) == 2 # epochs per step
    assert m._histories == m2._histories
    m2.step()
    assert m._histories != m2._histories
    assert m._histories == m2._histories[:2]

