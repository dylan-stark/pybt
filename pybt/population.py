from copy import copy

from pybt.member import Member
from pybt.model import ModelWrapper

class Population:
    def __init__(self, model, stopping_criteria=None, step_args={},
            eval_args={}):
        """Initialize a population with a model."""

        wrapped_model = ModelWrapper(model)
        self._members = [Member(wrapped_model, 0, stopping_criteria,
            step_args, eval_args)]

    def __len__(self):
        return len(self._members)

    def __str__(self):
        s = 'Population:\n'
        s += '\n'.join([str(m) for m in self._members])

        return s

    def train(self):
        """Train a population for some number of steps."""

        member = copy(self._members[0])
        while not member.done():
            member.step()
            member.eval()
            member = self._update(member)

        return self._best()

    def observations(self):
        return [m.observations() for m in self._members]

    def _best(self):
        return self._members[-1]._model._model

    def _update(self, member):
        self._members.append(member)
        return copy(self._members[-1])

