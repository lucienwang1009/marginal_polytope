import subprocess
import re

from logzero import logger
from contexttimer import Timer

from generator import MLNGenerator
from mln import MLN

class PartitionFunctionSolver(object):
    pass


class WFOMCSolver(PartitionFunctionSolver):
    def __init__(self):
        self.command = [
            'java', '-jar', 'forclift.jar', '-z',
            '--format-in', 'mln'
        ]
        self.mln_generator = MLNGenerator()
        self.pattern = re.compile(r'Z = exp\(([\d\.\-E]+)\)')

        self.calls = 0

    def solve(self, mln):
        """
        Solve the partition function problem for given MLN.
        Return ln(Z) where Z is the partition function.
        """
        self.calls += 1
        with self.mln_generator.generate(mln) as file_name:
            command = self.command + [file_name]
            logger.debug('command: %s', ' '.join(map(str, command)))
            with Timer() as t:
                result = subprocess.run(
                    command, stdout=subprocess.PIPE
                ).stdout.decode('utf-8')
            logger.info('elapsed time for WFOMC call: %s', t.elapsed)
            logger.debug('result: %s', result)
            res = re.findall(self.pattern, result)
            if not res or len(res) > 1:
                raise RuntimeError('Exception while running WFOMC: {}'.format(result))
            return float(res[0])


if __name__ == '__main__':
    mln = MLN(
        ['person'],
        ['friends(person,person)', 'smokes(person)'],
        ['smokes(x)', 'friends(x,y) ^ smokes(x) => smokes(y)'],
        2, [1, 1]
    )
    solver = WFOMCSolver()
    print(solver.solve(mln))
