import argparse
import os
import logging
import logzero
import pickle
import numpy as np

from logzero import logger
from contexttimer import Timer
from pracmln import MLN

from partition_func_solver import WFOMCSolver, ComplexWFOMCSolver
from solver import IterPolytopeSolver, DFTPolytopeSolver
# from utils import get_orthogonal_vector
from utils import plot_convex_hull

example_usage = '''Example:
python main.py -d person -p 'smokes(person);friends(person,person)' \\
    -f 'smokes(x);smokes(x) ^ friends(x,y) => smokes(y)' -s 2
'''


METHODS = {
    'iter': (IterPolytopeSolver, WFOMCSolver),
    'dft': (DFTPolytopeSolver, ComplexWFOMCSolver)
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='calculate marginal polytope of MLN',
        epilog=example_usage,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='mln file')
    parser.add_argument('--method', '-m', type=str, default='iter',
                        const='None', nargs='?', choices=METHODS.keys())
    parser.add_argument('--output_dir', '-o', type=str,
                        default='./check-points',
                        help="output convex hull pickle file in the formed of"
                             " scipy.spatial.ConvexHull and log file")
    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()
    return args


def construct_surrogate_polytope(mln, method='iter', verbose=False, max_vertices=None):
    log_level = logger.level
    if verbose:
        logzero.loglevel(logging.DEBUG)
    else:
        logzero.loglevel(logging.INFO)

    PolytopeSolver, PartitionFuncSolver = METHODS[method]

    mln_weights = [float(w) for w in mln.weights]

    def vertex_key_func(vertex):
        return np.dot(mln_weights, vertex.coordinate)
    with PartitionFuncSolver() as s:
        solver = PolytopeSolver(s, mln)
        try:
            with Timer() as t:
                if max_vertices is not None:
                    convex_hull = solver.get_convex_hull(max_vertices, vertex_key_func)
                else:
                    convex_hull = solver.get_convex_hull()
            logger.info('Total time for finding convex hull: %s', t.elapsed)
        except Exception as e:
            raise e
        finally:
            logger.info('num of call WFOMC: {}'.format(solver.solver.calls))

    # set back log level
    logzero.loglevel(log_level)
    return convex_hull


def construct_polytope(mln, method='iter', verbose=False):
    return construct_surrogate_polytope(mln, method, verbose)


if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.debug:
        logzero.loglevel(logging.DEBUG)
    else:
        logzero.loglevel(logging.INFO)
    logzero.logfile('{}/log.txt'.format(args.output_dir), mode='w')

    mln = MLN.load(args.input, grammar='StandardGrammar')
    convex_hull = construct_polytope(mln, args.method, args.debug)
    plot_convex_hull(convex_hull, '{}/polytope.png'.format(args.output_dir))
    with open('{}/convex_hull.pkl'.format(args.output_dir), 'wb') as f:
        pickle.dump(convex_hull, f)
