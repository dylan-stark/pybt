import logging

from copy import copy, deepcopy

logger = logging.getLogger(__name__)

class Member:
    """A member of the population.

    # Arguments:
        model: a wrapped model.
        model_id: an identifier for the model.
        stopping_criteria: some stopping criteria.
        ready_strategy: a means to determine readiness.
        step_args: arguments for each step, to be passed on to the model fit
            method.
        eval_args: arguments for model evaluation, to be passed on to the
            model evaluation method.
    """

    def __init__(self, model, model_id, stopping_criteria=None,
            ready_strategy=None, step_args={}, eval_args={}):
        logger.debug('Member(model={}, model_id={}, stopping_criteria={}, '
            'ready_strategy={}, step_args={}, eval_args={})'.format(model_id,
            model_id, stopping_criteria, ready_strategy, step_args, eval_args))

        self._model = model
        self._id = model_id

        self._step_args = {
            'epochs_per_step': step_args['epochs_per_step'],
            'fit_args': {
                'initial_epoch': step_args['fit_args'].get('initial_epoch', 0),
                'epochs': step_args['fit_args'].get('epochs', 0),
                'batch_size': step_args['fit_args'].get('batch_size', None),
                'x': step_args['fit_args'].get('x', None),
                'y': step_args['fit_args'].get('y', None),
                'validation_data': step_args['fit_args'].get('validation_data', None)
            }
        }

        self._evaluate_args = copy(eval_args)
        self._stopping_criteria = stopping_criteria
        self._ready_strategy = ready_strategy

        self._name = 'm{}'.format(model_id)

        self._t = 0
        self._observations = []

        self._p = None

        self.eval()

        logger.debug('Member(_) = {}'.format(self))

    def __copy__(self):
        logger.debug('copy({})'.format(self))

        m = Member(copy(self._model), self._id+1, self._stopping_criteria,
            self._ready_strategy, self._step_args, self._evaluate_args)
        m._observations = deepcopy(self._observations)
        m._t = self._t

        logger.debug('copy(_) = {}'.format(m))

        return m

    def __str__(self):
        s = 'M(p={}, t={}, e={})'.format(self._p, self._t,
            self._step_args['fit_args']['initial_epoch'])

        return s

    def step(self):
        logger.debug('step({})'.format(self))

        self._step_args['fit_args']['epochs'] = \
            self._step_args['fit_args']['initial_epoch'] + \
            self._step_args['epochs_per_step']

        obs = self._model.fit(self._step_args['fit_args'])
        obs['t'] = self._t
        self._observations.append(obs)

        self._step_args['fit_args']['initial_epoch'] += \
            self._step_args['epochs_per_step']

        self._t += 1
        self._p = None

        logger.debug('step(_) = {}'.format(self))

    def eval(self):
        logger.debug('eval({})'.format(self))
        logger.debug('_evaluate_args={}'.format(self._evaluate_args))

        loss, acc = self._model.eval(self._evaluate_args)
        self._p = acc

        logger.debug('eval(_) = {}'.format(self))

    def explore(self):
        logger.debug('explore({})'.format(self))

        self._model.explore()

    def done(self):
        logger.debug('done({})'.format(self))

        return self._stopping_criteria(self._step_args['fit_args'])

    def ready(self):
        logger.debug('ready({})'.format(self))

        status = self._ready_strategy(self._step_args['fit_args'])
        logging.debug('ready status = {}'.format(status))

        return status

    def observations(self):
        logger.debug('observations({})'.format(self))

        obs = {
            'observations': self._observations,
            'member': self._name
        }

        return obs

