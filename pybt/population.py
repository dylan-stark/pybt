import numpy as np

from pybt.member import Member

class Population:
    def __init__(self, models=None, step_args={}, eval_args={}):
        """Initialize the population."""
        if models==None:
            raise NotImplementedError
        elif isinstance(models, list):
            pass
        else:
            models = [models]

        self._members = [Member(m, step_args, eval_args) for m in models]

    def train(self, num_steps):
        """Train a population for some number of steps."""
        if num_steps <= 0:
            raise ValueError('num_steps must be positive integer')

        member = self._sample_from_population()
        for _ in range(num_steps):
            member.step()
            member.eval()

        return self._best()

    def _best(self):
        return self._members[0]._model

    def _sample_from_population(self):
        return np.random.choice(self._members)

