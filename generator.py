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
        fd = tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.mln',
            delete=False
        )
        content = ''
        for name, size in zip(mln.domain_name, mln.domain_size):
            items = list(map(str, range(size)))
            content += '{} = {{{}}}{}'.format(
                name,
                ', '.join(items),
                os.linesep
            )
        content += os.linesep
        for predicate in mln.predicates:
            content += predicate + os.linesep
        content += os.linesep

        world_size = mln.world_size
        logger.debug('world size: %s', world_size)
        for index, formula in enumerate(mln.formulas):
            # NOTE: mln can have complex weight
            formula_weight = mln.formula_weights[index]
            if isinstance(formula_weight, complex):
                weight_str = '{},{}'.format(
                    formula_weight.real,
                    formula_weight.imag
                )
            else:
                weight_str = formula_weight
            content += '{} {}{}'.format(
                weight_str,
                formula,
                os.linesep
            )
        fd.file.write(content)
        fd.close()
        try:
            yield fd.name
        finally:
            logger.debug('delete tmp file')
            os.remove(fd.name)


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
