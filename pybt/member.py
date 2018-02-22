from copy import copy

class Member:
    def __init__(self, model, step_args, eval_args):
        self._model = model
        self._fit_args = copy(step_args)
        self._evaluate_args = copy(eval_args)

        self._fit_args['initial_epoch'] = 0
        self._fit_args['epochs'] = 0
        self._epochs_per_step = 1

    def __copy__(self):
        m = Member(self._model, self._fit_args, self._evaluate_args)
        m._fit_args['initial_epoch'] = copy(self._fit_args['initial_epoch'])
        m._fit_args['epochs'] = copy(self._fit_args['epochs'])

        return m

    def __str__(self):
        s = '\t{} (t={})'.format(self._model, self._fit_args['initial_epoch'])

        return s

    def step(self):
        self._fit_args['epochs'] = \
            self._fit_args['initial_epoch'] + self._epochs_per_step
        self._model.fit(**self._fit_args)
        self._fit_args['initial_epoch'] += self._epochs_per_step

    def eval(self):
        loss, acc = self._model.evaluate(**self._evaluate_args)
        self._p = acc

