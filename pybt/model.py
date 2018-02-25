class ModelWrapper:
    def __init__(self, model):
        self._model = model
        self.metrics_names = model.metrics_names

    def __str__(self):
        s = 'Model ({})'.format(self._model)
        return s

    def fit(self, fit_args):
        history = self._model.fit(**fit_args)

        h = history.history
        h['epochs'] = list(range(fit_args['initial_epoch'], fit_args['epochs']))

        return h

    def evaluate(self, eval_args):
        loss, acc = self._model.evaluate(**eval_args)
        return loss, acc

