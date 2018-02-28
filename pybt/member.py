from copy import copy, deepcopy

class StopAfter:
    def __init__(self, epochs):
        self._epochs = epochs

    def __call__(self, fit_args):
        return self._epochs < (fit_args['initial_epoch'] + 1)

class Member:
    def __init__(self, model, model_id, stopping_criteria=StopAfter(epochs=10),
            step_args={}, eval_args={}):
        self._model = model
        self._id = model_id

        self._step_args = {
            'epochs_per_step': step_args['epochs_per_step'],
            'fit_args': {
                'initial_epoch': step_args['fit_args'].get('initial_epoch', 0),
                'epochs': step_args['fit_args'].get('epochs', 0),
                'x': step_args['fit_args']['x'],
                'y': step_args['fit_args']['y']
            }
        }

        self._evaluate_args = copy(eval_args)
        self._stopping_criteria = stopping_criteria

        self._name = 'm{}'.format(model_id)

        self._t = 0
        self._observations = []

        self.eval()

    def __copy__(self):
        m = Member(self._model, self._id+1, self._stopping_criteria,
            self._step_args, self._evaluate_args)
        m._observations = deepcopy(self._observations)
        m._t = self._t
        m._step_args['fit_args']['initial_epoch'] = \
            self._step_args['fit_args']['initial_epoch']
        m._step_args['fit_args']['epochs'] = \
            self._step_args['fit_args']['epochs']

        return m

    def __str__(self):
        s = '\t{} ({} @ t={}, e={})\n'.format(self._model,
                self._p, self._t, self._step_args['fit_args']['initial_epoch'])

        return s

    def step(self):
        self._step_args['fit_args']['epochs'] = \
            self._step_args['fit_args']['initial_epoch'] + \
            self._step_args['epochs_per_step']

        obs = self._model.fit(self._step_args['fit_args'])
        obs['t'] = self._t
        self._observations.append(obs)

        self._step_args['fit_args']['initial_epoch'] += \
            self._step_args['epochs_per_step']

        self._t += 1

    def eval(self):
        loss, acc = self._model.evaluate(self._evaluate_args)
        self._p = acc

    def done(self):
        return self._stopping_criteria(self._step_args['fit_args'])

    def observations(self):
        obs = {
            'observations': self._observations,
            'member': self._name
        }

        return obs

