import numpy as np
import pandas as pd

class ModelWrapper:
    def __init__(self, model, model_id):
        self._model = model
        self._id = model_id

        self.metrics_names = model.metrics_names

    def __str__(self):
        s = 'PyBT Model {}'.format(self._id)
        return s

    def fit(self, fit_args):
        history = self._model.fit(**fit_args)

        df = self._history_as_data_frame(history.history,
            fit_args['initial_epoch'], fit_args['epochs'])

        return df

    def evaluate(self, eval_args):
        loss, acc = self._model.evaluate(**eval_args)
        return loss, acc

    def _history_as_data_frame(self, history, start_epoch, stop_epoch):
        history['epoch'] = range(start_epoch, stop_epoch)
        df =  pd.DataFrame(history)

        return df

