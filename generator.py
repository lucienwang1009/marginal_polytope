import os
import tempfile

from logzero import logger
from contextlib import contextmanager

from mln import MLN

class MLNGenerator(object):
    def __init__(self):
        super().__init__()

    @contextmanager
    def generate(self, mln):
        _, file_name = tempfile.mkstemp()
        with open(file_name, 'w') as fd:
            for name, size in zip(mln.domain_name, mln.domain_size):
                fd.write('{} = {{{}}}{}'.format(
                    name,
                    ', '.join(map(str, range(size))),
                    os.linesep
                ))
            fd.write(os.linesep)
            for predicate in mln.predicates:
                fd.write(predicate + os.linesep)
            fd.write(os.linesep)

            world_size = mln.world_size
            logger.debug('world size: %s', world_size)
            for index, formula in enumerate(mln.formulas):
                fd.write('{} {}{}'.format(
                    mln.formula_weights[index],
                    formula,
                    os.linesep
                ))
        try:
            yield file_name
        finally:
            logger.debug('delete tmp file')
            os.remove(file_name)


if __name__ == '__main__':
    mln = MLN(
        ['person'],
        ['friends(person,person)', 'smokes(person)'],
        ['smokes(x)', 'friends(x,y) ^ smokes(x) => smokes(y)'],
        4, [1, 1]
    )
    generator = MLNGenerator()
    with generator.generate(mln) as file_name:
        with open(file_name, 'r') as f:
            for line in f.readlines():
                print(line.strip())
