import numpy as np
import math
import scipy
import seaborn as sns

from itertools import product
from logzero import logger
from matplotlib import pyplot as plt
from contexttimer import Timer

from mln import MLN
from polytope import IntegralConvexHull, Vertex
from utils import normalize_norm


class PolytopeSolver(object):
    def __init__(self, partition_func_solver, mln):
        super().__init__()
        self.solver = partition_func_solver

        self.dimension = len(mln.formulas)
        self.mln = mln

    def get_convex_hull(self):
        raise NotImplementedError


class IterPolytopeSolver(PolytopeSolver):
    def __init__(self, partition_func_solver, mln):
        super().__init__(partition_func_solver, mln)
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
        new_facet_norm = self._get_new_norm(vertex)
        logger.debug('new facet norm: %s', new_facet_norm)
        new_facet_b = self.get_b(new_facet_norm)
        logger.debug('new facet b: %s', new_facet_b)
        # this possible vertex is a true vertex
        if new_facet_norm.dot(vertex.coordinate) <= new_facet_b:
            logger.debug('possible vertex %s is a true vertex or in polytope', vertex.coordinate)
            return False
        with Timer() as t:
            self.convex_hull.add_facet(new_facet_norm, new_facet_b)
        logger.info('elapsed time for add facet: %s', t.elapsed)
        return True

    def get_convex_hull(self):
        self._init_convex_hull()
        logger.debug("Initial convex hull: \n%s", self.convex_hull)
        add_new_vertex = True
        while add_new_vertex:
            add_new_vertex = False
            for vertex in self.convex_hull.vertices.values():
                if vertex in self.visited:
                    continue
                self.visited.add(vertex)
                if self._find_new_vertices(vertex):
                    add_new_vertex = True
                    break
        return self.convex_hull.to_scipy_convex_hull()

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

    def _set_mln_weights(self, norm_vector):
        self.mln.formula_weights = [
            2 * i * math.log(self.mln.world_size) for i in norm_vector
        ]

    def get_b(self, norm_vector):
        self._set_mln_weights(norm_vector)
        ln_Z = self.solver.solve(self.mln)
        logger.debug("ln(Z) = %s", ln_Z)
        b = (ln_Z / math.log(self.mln.world_size) - 1) / 2
        b_round = round(b)
        if abs(b - b_round) <= 1e-10:
            b = b_round
        b = math.ceil(b)
        logger.debug("find b = %s", b)
        return b


class DFTPolytopeSolver(PolytopeSolver):
    def __init__(self, partition_func_solver, mln):
        super().__init__(partition_func_solver, mln)
        self.eps = 1e-10

    def _get_M(self):
        n_vars = self.mln.formula_vars
        assert len(self.mln.domain_size) == 1
        logger.info(n_vars)
        M = [(self.mln.domain_size[0] ** n) + 1 for n in n_vars]
        return M

    def dft(self):
        """
        g(k) = WFOMC(Phi_k) / Z
        """
        self.mln.formula_weights = [complex(0) for i in self.mln.formulas]
        denominator_ln_Z = self.solver.solve(self.mln)
        M = self._get_M()
        res = []
        range_M = [np.arange(m) for m in M]
        for i, m in enumerate(M):
            range_M[i] = complex(0, -2 * math.pi / m) * range_M[i]
        for weights in product(*range_M):
            self.mln.formula_weights = weights
            ln_Z = self.solver.solve(self.mln)
            res.append(math.e ** (ln_Z - denominator_ln_Z))
        res = np.array(res, dtype=np.complex256).reshape(M)
        return res

    def counting_distribution(self):
        logger.debug('calculate DFT of counting distribution')
        dft = self.dft()
        # distribution are real values, ignore image
        dist = np.fft.ifftn(dft).astype(np.float128)
        return dist

    def plot(self, dist):
        dist[np.where(np.abs(dist) < self.eps)] = 0
        dist_log = np.log10(dist)
        dist_log[np.isinf(dist_log)] = dist_log[np.isfinite(dist_log)].min() - 1
        sns.heatmap(dist_log)
        plt.savefig('./dist.png')

    def get_convex_hull(self):
        dist = self.counting_distribution()
        # self.plot(dist)
        np.save('./dist.npy', dist)
        non_zero_points = np.array(np.where(dist > self.eps)).T
        logger.debug(non_zero_points)
        return scipy.spatial.ConvexHull(non_zero_points)
