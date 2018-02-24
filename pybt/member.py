from copy import copy, deepcopy

import numpy as np

class Member:
    def __init__(self, model, step_args, eval_args):
        self._model = model
        self._observations = []
        self._fit_args = copy(step_args)
        self._evaluate_args = copy(eval_args)

        self._fit_args['initial_epoch'] = 0
        self._fit_args['epochs'] = 0
        self._epochs_per_step = 2

        self.eval()

    def __copy__(self):
        m = Member(self._model, self._fit_args, self._evaluate_args)
        m._observations = deepcopy(self._observations)
        m._fit_args['initial_epoch'] = copy(self._fit_args['initial_epoch'])
        m._fit_args['epochs'] = copy(self._fit_args['epochs'])

        return m

    def __str__(self):
        s = '\t{} ({} @ t={})\n'.format(self._model,
                self._p, self._fit_args['initial_epoch'])
        if len(self._observations) > 0:
            s += '{}\n'.format(np.vstack(self._observations))

        return s

    def asarray(self):
        if len(self._observations) == 0:
            return np.array([self._p, None, None, None], dtype='float64')
        else:
            obs = np.concatenate(self._observations)
            return np.insert(obs, 0, self._p, axis=1)

    def step(self):
        self._fit_args['epochs'] = \
            self._fit_args['initial_epoch'] + self._epochs_per_step

        obs = self._model.fit(self._fit_args)
        self._observations.append(obs)

        self._fit_args['initial_epoch'] += self._epochs_per_step

    def eval(self):
        loss, acc = self._model.evaluate(self._evaluate_args)
        self._p = acc

