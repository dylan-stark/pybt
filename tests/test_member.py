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

