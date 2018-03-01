import logging

from copy import copy

class Population:
    """A collection of model members.

    # Arguments:
        member: an initial member for the population.
    """

    def __init__(self, member):
        self._logger = logging.getLogger(__name__)
        self._logger.debug('Population({})'.format(member))

        self._members = [member]

        self._logger.debug('Population(_) = {}'.format(self))

    def __iter__(self):
        return iter(self._members)

    def __len__(self):
        return len(self._members)

    def __str__(self):
        s = '[{}]'.format(', '.join([str(m) for m in self._members]))
        return s

    def get(self):
        member = sorted(self._members, key=lambda x: x._p)[-1]

        self._logger.debug('get() = {}"'.format(member))

        return copy(member)

    def put(self, x):
        self._logger.debug('put({})'.format(x))

        self._members.append(x)
        y = copy(self._members[-1])

        self._logger.info('population after put = {}'.format(self))

        self._logger.debug('put(_) = {}'.format(y))

        return y

