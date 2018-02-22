import numpy as np

class Population:
    def __init__(self, models=None):
        """Initialize the population."""
        if not models==None and not isinstance(models, list):
            models = [models]

        self._models = models

    def train(self, num_steps, step_args):
        """Train a population for some number of steps."""
        if num_steps <= 0:
            raise ValueError('num_steps must be positive integer')

        step_args['initial_epoch'] = 0
        epochs_per_step = 1

        member = self._sample_from_population()
        for _ in range(num_steps):
            step_args['epochs'] = step_args['initial_epoch'] + epochs_per_step
            self._step(member, **step_args)
            step_args['initial_epoch'] += epochs_per_step

        return self._models[0]

    def _sample_from_population(self):
        return np.random.choice(self._models)

    def _step(self, member, **step_args):
        model = member
        model.fit(**step_args)

