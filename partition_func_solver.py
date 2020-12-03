import subprocess
import re
import time

from logzero import logger
from contexttimer import Timer

from generator import MLNGenerator
from mln import MLN
from py4j.java_gateway import JavaGateway


class PartitionFunctionSolver(object):
    pass


class WFOMCSolver(PartitionFunctionSolver):
    def __init__(self):
        self.mln_generator = MLNGenerator()
        self.pattern = re.compile(r'exp\(([\d\.\-E]+)\)')

        self.calls = 0

    def __enter__(self):
        self.process = self.start_forclift()
        time.sleep(0.5)
        return self

    def __exit__(self, type, value, traceback):
        self.stop_forclift()

    def start_forclift(self):
        command = [
            'java', '-jar', 'forclift.jar', '--gateway'
        ]
        return subprocess.Popen(command)

    def solve(self, mln):
        """
        Solve the partition function problem for given MLN.
        Return ln(Z) where Z is the partition function.
        """
        self.calls += 1
        gateway = JavaGateway()
        with self.mln_generator.generate(mln) as file_name:
            with Timer() as t:
                result = gateway.entry_point.WFOMC(file_name)
            logger.info('elapsed time for WFOMC call: %s', t.elapsed)
            logger.debug('result: %s', result)
            res = re.findall(self.pattern, result)
            if not res or len(res) > 1:
                raise RuntimeError('Exception while running WFOMC: {}'.format(result))
            return float(res[0])

    def stop_forclift(self):
        self.process.kill()


if __name__ == '__main__':
    mln = MLN(
        ['person'],
        ['friends(person,person)', 'smokes(person)'],
        ['smokes(x)', 'friends(x,y) ^ smokes(x) => smokes(y)'],
        2, [1, 1]
    )
    solver = WFOMCSolver()
    print(solver.solve(mln))
