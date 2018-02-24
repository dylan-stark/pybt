import numpy as np

class ModelWrapper:
    def __init__(self, model, model_id):
        self._model = model
        self._id = model_id

    def __str__(self):
        s = 'PyBT Model {}'.format(self._id)
        return s

    def fit(self, fit_args):
        history = self._model.fit(**fit_args)

        table = self._history_table(self._model, history,
            fit_args['initial_epoch'], fit_args['epochs'])

        return table

    def evaluate(self, eval_args):
        loss, acc = self._model.evaluate(**eval_args)
        return loss, acc

    def _history_table(self, model, history, start_epoch, stop_epoch):
        """Create a table of observations per epoch."""
        metric_table = [range(start_epoch, stop_epoch)] + \
            [history.history[k] for k in model.metrics_names]
        obs_table = np.array(metric_table, dtype='float64').transpose()

        return obs_table

