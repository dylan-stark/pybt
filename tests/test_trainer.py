from math import floor

import pytest

from pybt.trainer import Trainer
from pybt.policy.done import StopAfter
from pybt.policy.ready import ReadyAfter

from conftest import SimpleWrapper, LinearModel, QuadraticModel

def test_linear_unit_steps(linear_model):
    """Test that we can train unit-sized steps."""
    stop_after_epochs = 10
    ready_after_epochs = 1
    epochs_per_step = 1

    t = Trainer(linear_model,
        step_args={
            'epochs_per_step': epochs_per_step,
            'fit_args': {'x': {}, 'y': {}}},
        eval_args={'x': {}, 'y': {}},
        stopping_criteria=StopAfter(stop_after_epochs),
        ready_strategy=ReadyAfter(ready_after_epochs))
    model, score = t.train()

    assert score == 10
    assert model.evaluate({}) == (-10, 10)
    assert len(t.observations()) == (stop_after_epochs // epochs_per_step) + 1

def test_quadratic_unit_steps():
    """Test that we can train unit-sized steps."""
    import random
    random.seed(0)

    stop_after_epochs = 10
    ready_after_epochs = 1
    epochs_per_step = 1

    t = Trainer(SimpleWrapper(QuadraticModel()),
        step_args={
            'epochs_per_step': epochs_per_step,
            'fit_args': {'x': {}, 'y': {}}},
        eval_args={'x': {}, 'y': {}},
        stopping_criteria=StopAfter(stop_after_epochs),
        ready_strategy=ReadyAfter(ready_after_epochs))
    model, score = t.train()

    assert floor(score) == 54
    loss, acc = model.evaluate({})

def test_linear_small_steps():
    """Test that we can train multi-epoch steps."""
    stop_after_epochs = 30
    ready_after_epochs = 1
    epochs_per_step = 3

    t = Trainer(SimpleWrapper(LinearModel()),
        step_args={
            'epochs_per_step': epochs_per_step,
            'fit_args': {'x': {}, 'y': {}}},
        eval_args={'x': {}, 'y': {}},
        stopping_criteria=StopAfter(stop_after_epochs),
        ready_strategy=ReadyAfter(ready_after_epochs))
    model, score = t.train()

    assert score == 30
    assert model.evaluate({}) == (-30, 30)
    assert len(t.observations()) == (stop_after_epochs // epochs_per_step) + 1

def test_linear_small_steps_when_ready():
    """Test that we can train multi-epoch steps."""
    stop_after_epochs = 30
    ready_after_epochs = 6
    epochs_per_step = 3

    t = Trainer(SimpleWrapper(LinearModel()),
        step_args={
            'epochs_per_step': epochs_per_step,
            'fit_args': {'x': {}, 'y': {}}},
        eval_args={'x': {}, 'y': {}},
        stopping_criteria=StopAfter(stop_after_epochs),
        ready_strategy=ReadyAfter(ready_after_epochs))
    model, score = t.train()

    assert score == 30
    assert model.evaluate({}) == (-30, 30)
    assert len(t.observations()) == (stop_after_epochs // epochs_per_step) + 1

def test_multiple_train_calls():
    stop_after_epochs = 30
    ready_after_epochs = 1
    epochs_per_step = 3

    t = Trainer(SimpleWrapper(LinearModel()),
        step_args={
            'epochs_per_step': epochs_per_step,
            'fit_args': {'x': {}, 'y': {}}},
        eval_args={'x': {}, 'y': {}},
        stopping_criteria=StopAfter(stop_after_epochs),
        ready_strategy=ReadyAfter(ready_after_epochs))

    model, score = t.train()
    assert score == 30
    assert model.evaluate({}) == (-30, 30)
    assert len(t.observations()) == (stop_after_epochs // epochs_per_step) + 1

    # Nothing should happen because the done criteria has already been met.
    m2, s2 = t.train()
    assert s2 == 30

@pytest.mark.skip('need to deprecate keras model')
def test_truncation(keras_model):
    """After 5 steps, we should swap the current member with one in the
    top 25%."""

    from pybt.policy.exploit import Truncation
    from pybt.policy.ready import ReadyAfter
    from pybt.policy.done import StopAfter

    t = Trainer(model=SimpleWrapper(keras_model),
        stopping_criteria=StopAfter(5),
        ready_strategy=ReadyAfter(5),
        exploit_strategy=Truncation(upper=.4, lower=.4),
        step_args={'epochs_per_step': 1, 'fit_args': {'x': {}, 'y':{}}})
    _, score = t.train()

    assert score in [84., 90.]

