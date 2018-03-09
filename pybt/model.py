"""Models: utilties for working with user models.
"""

import logging

from abc import ABC, abstractmethod
from copy import copy

import numpy as np

from keras.models import clone_model

logger = logging.getLogger(__name__)

class ModelWrapper(ABC):
    """Abstract wrapper class for user models.

    User model wrappers must implement `__copy__()`, `_clone()`, `fit()`,
    `eval()`, and `explore()` methods. The `clone()` method should return a
    new copy of the model. The `fit()` and `eval()` methods should train and
    evaluate the model, resp.  And the `explore()` method should "perturb" the
    model hyperparameters in some way that is meaningful to the user.

    Note that the user's model is copied on initialization. So any changes to
    the model outside of PyBT training should be independent of the traning
    process.

    # Arguments
        model: a user model.
    """
    def __init__(self, model, **kwargs):
        logger.debug('ModelWrapper(model={}, **kwargs={})'.format(model,
            kwargs))

        self._kwargs = kwargs

        self._model = self._clone(model)

    @abstractmethod
    def __copy__(self):
        pass

    @abstractmethod
    def fit(self, fit_args):
        pass

    @abstractmethod
    def eval(self, eval_args):
        pass

    @abstractmethod
    def explore(self):
        pass

    @abstractmethod
    def _clone(self, model):
        pass

class KerasModelWrapper(ModelWrapper):
    """Keras model wrapper.

    # Arguments:
        **kwargs: model compilation arguments (e.g., optimizer, loss, etc.).
    """

    def __copy__(self):
        logger.debug('copy({})'.format(self))

        return KerasModelWrapper(self._model, **self._kwargs)

    def fit(self, fit_args):
        logger.debug('fit(fit_args={})'.format(fit_args))

        history = self._model.fit(**fit_args)

        # Add epoch range to history
        h = history.history
        h['epochs'] = list(range(fit_args['initial_epoch'], fit_args['epochs']))

        return h

    def eval(self, eval_args):
        logger.debug('eval(eval_args={})'.format(eval_args))

        loss, acc = self._model.evaluate(**eval_args)
        return loss, acc

    def explore(self):
        logger.debug('explore()')

        import keras.backend as K

        weights = [np.random.permutation(w.flat).reshape(w.shape) \
                   for w in self._model.get_weights()]
        self._model.set_weights(weights)

    def _clone(self, model):
        logger.debug('_clone(model={})'.format(model))

        x = clone_model(model)
        x.set_weights(model.get_weights())
        x.compile(**self._kwargs)

        return x

