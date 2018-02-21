class Population:
    def __init__(self, models=None):
        """Initialize the population."""
        if not models==None and not isinstance(models, list):
            models = [models]

        self._models = models

    def train(self, num_steps):
        """Train a population for some number of steps."""
        if num_steps <= 0:
            raise ValueError('num_steps must be positive integer')

        return self._models

