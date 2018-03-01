import logging

from copy import copy

from pybt.population import Population
from pybt.member import Member

from pybt.policy.exploit import Truncation
from pybt.policy.ready import ReadyAfter
from pybt.policy.done import StopAfter

class Trainer:
    def __init__(self, model, stopping_criteria=StopAfter(10),
            ready_strategy=ReadyAfter(2), exploit_strategy=Truncation(),
            step_args={}, eval_args={}):
        """Initialize a population with a model."""
        self._logger = logging.getLogger(__name__)
        self._logger.debug('Trainer({}, {}, {})'.format(
            stopping_criteria, ready_strategy, exploit_strategy))

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
        """Train a population for some number of steps."""
        self._logger.info('starting training')

        member = self._members.get()
        self._logger.info('starting with {} from {}'.format(member,
            self._members))
        while not member.done():
            member.step()
            member.eval()
            self._logger.info('after step+eval: {}'.format(member))
            if member.ready():
                candidate = self.exploit(member)
                if candidate != member:
                    self._logger.info('using new candidate')
                    # Save member to population before overwriting
                    self._members.put(member)
                    member = candidate
                    self._logger.info('member now = {}'.format(member))
                    member = self.explore(member)
                    self._logger.info('after explore = {}'.format(member))
                    member.eval()
                    self._logger.info('member eval = {}'.format(member))
                else:
                    self._logger.info('discarding candidate')
            member = self._update(member)

        return self._best()

    def exploit(self, member):
        self._logger.debug('exploit({})'.format(member))
        candidate = self._exploit_strategy(member, self._members)
        self._logger.debug('exploit({}) = {}'.format(member, candidate))

        return candidate

    def explore(self, member):
        member.explore()
        self._logger.debug('after explore = {}'.format(member))
        return member

    def observations(self):
        return [m.observations() for m in self._members]

    def _best(self):
        m = max(self._members, key=lambda x: x._p)
        return m._model._model, m._p

    def _update(self, member):
        self._logger.debug('update({})'.format(member))
        return self._members.put(member)

