class Population:
    def __init__(self, models=None):
        """Initialize the population.
        """
        if not models==None and not isinstance(models, list):
            models = [models]

        self._models = models

    def train(self, num_steps=None):
        return self._models

