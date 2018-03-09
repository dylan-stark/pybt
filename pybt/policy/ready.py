"""Ready policies.
"""

import logging

logger = logging.getLogger(__name__)

class ReadyAfter:
    """Flag model as ready after so many epochs.

    # Arguments
        epochs: number of epochs to go before being ready.
    """

    def __init__(self, epochs):
        logger.debug('ReadyAfter(epochs={})'.format(epochs))

        self._epochs = epochs
        self._last_ready = 0

    def __call__(self, fit_args):
        logger.debug('ReadyAfter()(fit_args={})'.format(fit_args))

        if fit_args['initial_epoch'] - self._last_ready >= self._epochs:
            self._last_ready = fit_args['initial_epoch']
            return True
        else:
            return False

    def __str__(self):
        s = 'ReadyAfter(epochs={})'.format(self._epochs)
        return s

