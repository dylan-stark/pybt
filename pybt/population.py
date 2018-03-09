import logging

from copy import copy

logger = logging.getLogger(__name__)

class Population:
    """A collection of model members.

    # Arguments:
        member: an initial member for the population.
    """

    def __init__(self, member):
        logger.debug('Population(member={})'.format(member))

        self._members = [member]

        logger.debug('Population(_) = {}'.format(self))

    def __iter__(self):
        return iter(self._members)

    def __len__(self):
        return len(self._members)

    def __str__(self):
        s = '[{}]'.format(', '.join([str(m) for m in self._members]))
        return s

    def get(self):
        logger.debug('get()')

        member = sorted(self._members, key=lambda x: x._p)[-1]

        logger.debug('get() = {}"'.format(member))

        return copy(member)

    def put(self, x):
        logger.debug('put(x={})'.format(x))

        self._members.append(x)
        y = copy(self._members[-1])

        logger.debug('population after put = {}'.format(self))
        logger.debug('put(_) = {}'.format(y))

        return y

