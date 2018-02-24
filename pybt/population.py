from copy import copy

import numpy as np

from pybt.member import Member
from pybt.model import ModelWrapper

class Population:
    def __init__(self, models=None, step_args={}, eval_args={}):
        """Initialize the population."""
        if models==None:
            raise NotImplementedError
        elif isinstance(models, list):
            pass
        else:
            models = [models]

        models = [ModelWrapper(m, i) for i, m in enumerate(models)]
        self._members = [Member(m, step_args, eval_args) for m in models]

    def __len__(self):
        return len(self._members)

    def __str__(self):
        s = 'Population:\n'
        s += '\n'.join([str(m) for m in self._members])

        return s

    def asarray(self):
        """Compile recorded observations across the population."""
        #return self._members[0].asarray()
        return np.vstack([m.asarray() for m in self._members])

    def train(self, num_steps):
        """Train a population for some number of steps."""
        if num_steps <= 0:
            raise ValueError('num_steps must be positive integer')

        member = self._sample_from_population()
        for _ in range(num_steps):
            member.step()
            member.eval()
            self._update(member)

        return self._best()

    def _best(self):
        return self._members[-1]._model

    def _sample_from_population(self):
        return copy(np.random.choice(self._members))

    def _update(self, member):
        self._members.append(copy(member))

