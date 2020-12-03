import argparse
import numpy as np
import math
import logging
import logzero

from logzero import logger
from itertools import combinations, product

from mln import MLN
from partition_func_solver import WFOMCSolver
from polytope import IntegralConvexHull, Vertex
# from utils import get_orthogonal_vector
from utils import normalize_norm
from contexttimer import Timer

example_usage = '''Example:
python main.py -d person -p 'smokes(person);friends(person,person)' \\
    -f 'smokes(x);smokes(x) ^ friends(x,y) => smokes(y)' -s 2
'''


def parse_args():
    parser = argparse.ArgumentParser(
        description='calculate marginal polytope of MLN',
        epilog=example_usage,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--domain_name', '-d', type=str, required=True,
                        help='domain names, split by semicolon')
    parser.add_argument('--predicates', '-p', type=str, required=True,
                        help='predicates, split by semicolon')
    parser.add_argument('--formulas', '-f', type=str, required=True,
                        help='formulas, split by semicolon')
    parser.add_argument('--domain_size', '-s', type=str, required=True,
                        help='domain size, if multiple, split by semicolon')
    parser.add_argument('--log_file', '-l', type=str, default='log.txt')
    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()
    return args


class PolytopeSolver(object):
    def __init__(self, partition_func_solver, domain_name,
                 predicates, formulas, domain_size):
        super().__init__()
        self.solver = partition_func_solver
        self.domain_name = domain_name
        self.predicates = predicates
        self.formulas = formulas
        self.domain_size = domain_size

        self.dimension = len(self.formulas)
        self.vertices = []
        self.visited = set()
        self.convex_hull = None
        self.norm_boundary = None

    def _get_integers_points(self, vertex):
        # find all feasible integer points on each hyperplane
        points = []

        # d points in d dimension form a hyperplane
        for i in range(self.dimension):
            points.append(self.convex_hull.get_feasible_point_on_facet(
                vertex.facets_norm[i],
                vertex.facets_intercept[i],
                vertex.coordinate,
                handler=lambda x: not np.all(x == vertex.coordinate)
            ))
        return points

    def _norm_in_boundary(self, norm, facets_norm):
        if np.all(norm >= -self.norm_boundary) and \
                np.all(norm <= self.norm_boundary):
            return True
        return False

    def _get_new_norm(self, vertex):
        """
        Find the norm of new hyperplane, which is not parallel to
        the facet of the given cone
        """
        return np.array(normalize_norm(np.sum(vertex.facets_norm, axis=0)))

    def _find_new_vertices(self, vertex):
        logger.debug('try to find new vertices based on %s', vertex)
        vertices = []
        new_facet_norm = self._get_new_norm(vertex)
        logger.debug('new facet norm: %s', new_facet_norm)
        new_facet_b = self.get_b(new_facet_norm)
        logger.debug('new facet b: %s', new_facet_b)
        # this possible vertex is a true vertex
        if new_facet_norm.dot(vertex.coordinate) == new_facet_b:
            logger.debug('possible vertex %s is a true vertex', vertex.coordinate)
            self.visited.add(vertex)
            return vertices
        with Timer() as t:
            vertices = self.convex_hull.add_facet(new_facet_norm, new_facet_b)
        logger.info('elapsed time for add facet: %s', t.elapsed)
        return vertices

    def get_convex_hull(self):
        self._init_convex_hull()
        logger.debug("Initial convex hull: \n%s", self.convex_hull)
        self.vertices = [v for v in self.convex_hull.get_vertices()]
        for vertex in self.vertices:
            if vertex in self.visited or \
                    self.convex_hull.get_vertex(vertex.coordinate) is None:
                continue
            vertices = self._find_new_vertices(vertex)
            self.vertices.extend(vertices)
        return self.convex_hull

    def _init_convex_hull(self):
        # find all possible initial vertices
        vertices = []
        bases = np.diag([1] * self.dimension).astype(np.float)
        min_max = np.zeros([self.dimension, 2]).astype(np.float)
        # the vertices of cube is possible vertices
        for i in range(self.dimension):
            b = self.get_b(-bases[i])
            min_max[i][0] = -b
            b = self.get_b(bases[i])
            min_max[i][1] = b
        for ind in product(*([[0, 1]] * self.dimension)):
            coordinate = np.array(
                [min_max[i][ind[i]] for i in range(self.dimension)],
            )
            sign = np.array(ind) * 2 - 1
            vertex = Vertex(
                coordinate,
                (bases * sign).astype(np.int32),
                (coordinate * sign).astype(np.int32)
            )
            vertices.append(vertex)
        self.convex_hull = IntegralConvexHull(self.dimension, vertices)

    def _construct_mln(self, norm_vector):
        mln = MLN(
            self.domain_name,
            self.predicates,
            self.formulas,
            self.domain_size
        )
        mln.formula_weights = [
            i * math.log(mln.world_size) for i in norm_vector
        ]
        return mln

    def get_b(self, norm_vector):
        mln = self._construct_mln(norm_vector)
        ln_Z = self.solver.solve(mln)
        logger.debug("ln(Z) = %s", ln_Z)
        b = ln_Z / math.log(mln.world_size)
        b_floor = math.floor(b)
        if abs(b - b_floor) < 1e-7:
            b = b_floor - 1
        else:
            b = b_floor
        logger.debug("find b = %s", b)
        return b


if __name__ == '__main__':
    args = parse_args()
    if args.debug:
        logzero.loglevel(logging.DEBUG)
    else:
        logzero.loglevel(logging.INFO)
    logzero.logfile(args.log_file, mode='w')
    with WFOMCSolver() as s:
        solver = PolytopeSolver(
            s, args.domain_name.split(';'),
            args.predicates.split(';'),
            args.formulas.split(';'),
            list(map(int, args.domain_size.split(';')))
        )
        try:
            convex_hull = solver.get_convex_hull()
        except Exception as e:
            raise e
        finally:
            logger.info('num of call WFOMC: {}'.format(solver.solver.calls))
    logger.info(convex_hull)
    logger.info('num of call WFOMC: {}'.format(solver.solver.calls))
    if convex_hull.dimension <= 3:
        convex_hull.show('./polytope.png')
