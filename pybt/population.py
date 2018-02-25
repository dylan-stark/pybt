from copy import copy

import pandas as pd

from pybt.member import Member
from pybt.model import ModelWrapper

class Population:
    def __init__(self, model, step_args={}, eval_args={}):
        """Initialize a population with a model."""

        wrapped_model = ModelWrapper(model)
        self._members = [Member(wrapped_model, 0, step_args, eval_args)]

    def __len__(self):
        return len(self._members)

    def __str__(self):
        s = 'Population:\n'
        s += '\n'.join([str(m) for m in self._members])

        return s

    def train(self, num_steps):
        """Train a population for some number of steps."""
        if num_steps <= 0:
            raise ValueError('num_steps must be positive integer')

        member = copy(self._members[0])
        for _ in range(num_steps):
            member.step()
            member.eval()
            member = self._update(member)

        return self._best()

    def as_data_frame(self):
        """Compile recorded observations across the population."""
        return pd.concat([m.as_data_frame() for m in self._members],
            ignore_index=True)

    def _best(self):
        return self._members[-1]._model._model

    def _update(self, member):
        self._members.append(member)
        return copy(self._members[-1])

