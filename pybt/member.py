from copy import copy, deepcopy

class Member:
    def __init__(self, model, model_id, step_args, eval_args):
        self._model = model
        self._id = model_id
        self._fit_args = copy(step_args)
        self._evaluate_args = copy(eval_args)

        self._name = 'm{}'.format(model_id)

        self._t = 0
        self._observations = []

        self._fit_args['initial_epoch'] = 0
        self._fit_args['epochs'] = 0
        self._epochs_per_step = 2

        self.eval()

    def __copy__(self):
        m = Member(self._model, self._id+1, self._fit_args, self._evaluate_args)
        m._observations = deepcopy(self._observations)
        m._t = self._t
        m._fit_args['initial_epoch'] = copy(self._fit_args['initial_epoch'])
        m._fit_args['epochs'] = copy(self._fit_args['epochs'])

        return m

    def __str__(self):
        s = '\t{} ({} @ t={}, e={})\n'.format(self._model,
                self._p, self._t, self._fit_args['initial_epoch'])

        return s

    def step(self):
        self._fit_args['epochs'] = \
            self._fit_args['initial_epoch'] + self._epochs_per_step

        obs = self._model.fit(self._fit_args)
        obs['t'] = self._t
        self._observations.append(obs)

        self._fit_args['initial_epoch'] += self._epochs_per_step

        self._t += 1

    def eval(self):
        loss, acc = self._model.evaluate(self._evaluate_args)
        self._p = acc

    def observations(self):
        obs = {
            'observations': self._observations,
            'member': self._name
        }

        return obs

