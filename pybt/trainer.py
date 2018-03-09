import logging

from copy import copy

from pybt.population import Population
from pybt.member import Member

from pybt.policy.exploit import Truncation
from pybt.policy.ready import ReadyAfter
from pybt.policy.done import StopAfter

logger = logging.getLogger(__name__)

class Trainer:
    """A model trainer.

    # Arguments:
        model: a wrapped model.
        stopping_criteria: some stopping criteria.
        ready_strategy: a means to determine readiness.
        exploit_strategy: a means to exploit a population.
        step_args: arguments for each step, to be passed on to the model fit
            method.
        eval_args: arguments for model evaluation, to be passed on to the
            model evaluation method.
    """

    def __init__(self, model, stopping_criteria=StopAfter(10),
            ready_strategy=ReadyAfter(2), exploit_strategy=Truncation(),
            step_args={}, eval_args={}):
        logger.debug('Trainer(model={}, stopping_criteria={}, '
            'ready_strategy={}, exploit_strategy={}, step_args={}, '
            'eval_args={})'.format(model, stopping_criteria, ready_strategy,
            exploit_strategy, step_args, eval_args))

        self._exploit_strategy = exploit_strategy

        member = Member(model, 0, stopping_criteria,
            ready_strategy, step_args, eval_args)

        self._members = Population(member)

    def __len__(self):
        return len(self._members)

    def __str__(self):
        s = 'Population:\n'
        s += '\n'.join([str(m) for m in self._members])

        return s

    def train(self):
        logger.debug('train()')

        member = self._members.get()
        while not member.done():
            member.step()
            member.eval()
            if member.ready():
                candidate = self.exploit(member)
                if candidate != member:
                    self._members.put(member)
                    member = candidate
                    member = self.explore(member)
                    member.eval()
            member = self._update(member)

        return self._best()

    def exploit(self, member):
        logger.debug('exploit(member={})'.format(member))

        candidate = self._exploit_strategy(member, self._members)

        logger.debug('exploit(_) = {}'.format(candidate))

        return candidate

    def explore(self, member):
        logger.debug('explore(member={})'.format(member))

        member.explore()

        logger.debug('explore(_) = {}'.format(member))

        return member

    def observations(self):
        logger.debug('observations()')

        return [m.observations() for m in self._members]

    def _best(self):
        logger.debug('_best()')

        m = max(self._members, key=lambda x: x._p)
        return m._model._model, m._p

    def _update(self, member):
        logger.debug('update(member={})'.format(member))

        return self._members.put(member)

