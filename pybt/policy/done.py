"""Done policies.
"""

class StopAfter:
    """Stop after so many epochs.

    # Arguments:
        epochs: number of epochs until training is done.
    """

    def __init__(self, epochs):
        self._epochs = epochs

    def __call__(self, fit_args):
        return self._epochs < (fit_args['initial_epoch'] + 1)

    def __str__(self):
        s = 'StopAfter(epochs={})'.format(self._epochs)
        return s

