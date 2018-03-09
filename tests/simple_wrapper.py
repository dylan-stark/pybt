from copy import copy

from pybt.model import ModelWrapper

class SimpleWrapper(ModelWrapper):
    def __copy__(self):
        return SimpleWrapper(self._model)

    def _clone(self, model):
        return copy(model)

    def fit(self, fit_args):
        history = self._model.fit(**fit_args)

        h = history.history
        h['epochs'] = list(range(fit_args['initial_epoch'], fit_args['epochs']))

        return h

    def eval(self, eval_args):
        loss, acc = self._model.evaluate(**eval_args)
        return loss, acc

    def explore(self, fit_args):
        self._model.explore()

