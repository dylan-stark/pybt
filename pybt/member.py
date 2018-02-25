from copy import copy, deepcopy

import numpy as np
import pandas as pd

class Member:
    def __init__(self, model, model_id, step_args, eval_args):
        self._id = model_id
        self._name = 'm{}'.format(model_id)
        self._t = 0

        self._model = model
        self._fit_args = copy(step_args)
        self._evaluate_args = copy(eval_args)

        empty_obs = {k: [] for k in model.metrics_names}
        empty_obs['epoch'] = int()
        empty_obs['t'] = int()
        empty_df = pd.DataFrame(empty_obs)
        self._observations = [empty_df]

        self._fit_args['initial_epoch'] = 0
        self._fit_args['epochs'] = 0
        self._epochs_per_step = 2

        print('Initialized {}'.format(self._name))

        self.eval()

    def __copy__(self):
        m = Member(self._model, self._id+1, self._fit_args, self._evaluate_args)
        m._observations = deepcopy(self._observations)
        m._t = self._t
        print('Deep copying m{} to {} obs'.format(self._id, m._name))
        m._fit_args['initial_epoch'] = copy(self._fit_args['initial_epoch'])
        m._fit_args['epochs'] = copy(self._fit_args['epochs'])

        return m

    def __str__(self):
        s = '\t{} ({} @ t={}, e={})\n'.format(self._model,
                self._p, self._t, self._fit_args['initial_epoch'])
        if len(self._observations) > 0:
            s += '{}\n'.format(np.vstack(self._observations))

        return s

    def step(self):
        print('Stepping {}'.format(self._name))

        self._fit_args['epochs'] = \
            self._fit_args['initial_epoch'] + self._epochs_per_step

        obs = self._model.fit(self._fit_args)
        obs['t'] = self._t
        self._observations.append(obs)
        print('num obs: {}'.format(len(self._observations)))

        self._fit_args['initial_epoch'] += self._epochs_per_step

        self._t += 1

    def eval(self):
        loss, acc = self._model.evaluate(self._evaluate_args)
        self._p = acc

    def as_data_frame(self):
        df = pd.concat(self._observations, ignore_index=True)
        df['model'] = self._name

        return df

