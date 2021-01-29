import numpy as np
import sys
import pickle
import heapq

from logzero import logger
from scipy.optimize import linprog
from matplotlib import pyplot as plt
from scipy.spatial import ConvexHull as scipy_convex_hull
from itertools import product
from contexttimer import Timer
from fractions import Fraction

from utils import coordinate2str, get_hyperplane, get_integral_vec, cartesian_product


class Facet(object):
    def __init__(self):
        super().__init__()
        self.norm_vector = []
        self.intercept = 0

    def __str__(self):
        return '{} * x <= {}'.format(self.norm_vector, self.intercept)


class Vertex(object):
    def __init__(self, coordinate, facets_norm, facets_intercept):
        super().__init__()
        self.coordinate = np.array(coordinate)
        self.facets_norm = np.array(facets_norm)
        self.facets_intercept = np.array(facets_intercept)

    def within_boundary(self, x):
        return np.all(self.facets_norm.dot(x) <= self.facets_intercept)

    def update_facet(self, norms, intercepts):
        """
        Add facet and remove redundant facets
        """
        self.facets_norm = np.concatenate(
            (self.facets_norm, norms)
        )
        self.facets_intercept = np.concatenate(
            (self.facets_intercept, intercepts)
        )
        unique_facets = np.unique(
            np.hstack((
                self.facets_norm,
                self.facets_intercept.reshape(self.facets_norm.shape[0], 1)
            )),
            axis=0
        )
        self.facets_norm = unique_facets[:, :-1]
        self.facets_intercept = unique_facets[:, -1]
        logger.debug('before prune redundant facets: \n%s', self)

        irredundant_facets_norm = []
        irredundant_facets_intercept = []
        for i in range(self.facets_norm.shape[0]):
            c = -self.facets_norm[i, :]
            A = np.vstack((
                self.facets_norm[:i, :],
                self.facets_norm[i + 1:, :]
            ))
            b = np.hstack((
                self.facets_intercept[:i],
                self.facets_intercept[i + 1:]
            ))
            try:
                res = linprog(c, A, b, bounds=(None, None))
            except ValueError:
                irredundant_facets_norm.append(self.facets_norm[i, :])
                irredundant_facets_intercept.append(self.facets_intercept[i])
                continue
            if -res.fun - self.facets_intercept[i] > 1e-5:
                irredundant_facets_norm.append(self.facets_norm[i, :])
                irredundant_facets_intercept.append(self.facets_intercept[i])
        self.facets_norm = np.array(irredundant_facets_norm)
        self.facets_intercept = np.array(irredundant_facets_intercept)
        logger.debug('after prune redundant facets: \n%s', self)

    def __str__(self):
        string = 'Coordinate: {}\n'.format(self.coordinate)
        string += 'Facets: \n{} * x <= {}\n'.format(self.facets_norm, self.facets_intercept)
        return string

    def __eq__(self, other):
        return coordinate2str(self.coordinate) == coordinate2str(other.coordinate)

    def __hash__(self):
        return hash(coordinate2str(self.coordinate))


class ConvexHull(object):
    def __init__(self, dimension):
        super().__init__()
        self.dimension = dimension
        # coordinate => vertex
        self.vertices = {}
        self.facets = []

    def contains(self, x):
        for vertex in self.vertices.values():
            if not vertex.within_boundary(x):
                return False
        return True

    def add_vertex(self, vertex):
        v = self.get_vertex(vertex.coordinate)
        if v is not None:
            logger.debug('Duplicate vertex: \n%s\n%s', vertex, v)
            v.update_facet(vertex.facets_norm,
                           vertex.facets_intercept)
        else:
            self.vertices[coordinate2str(vertex.coordinate)] = vertex

    def get_vertex(self, coordinate):
        return self.vertices.get(coordinate2str(coordinate), None)

    def get_vertices(self):
        return self.vertices.values()

    def remove_vertex(self, vertex):
        if self.get_vertex(vertex.coordinate) is None:
            raise RuntimeError('Unable to remove vertex {}'.format(vertex.coordinate))
        del self.vertices[coordinate2str(vertex.coordinate)]

    def to_scipy_convex_hull(self):
        corners = []
        for v in self.vertices.values():
            corners.append(v.coordinate)
        corners = np.array(corners)
        hull = scipy_convex_hull(corners)
        return hull

    def get_feasible_point_on_facet(
            self, norm, intercept, initial_point, handler=None
    ):
        """
        Get the feasible integer points on the given facet
        """
        sys.setrecursionlimit(10000)

        visited = set()
        res = []
        directions = np.zeros(
            [2 * len(initial_point), len(initial_point)],
            dtype=np.int32
        )
        for i in range(len(initial_point)):
            directions[2 * i][i] = 1
            directions[2 * i + 1][i] = -1

        def dfs(point):
            # nonlocal res, visited
            if not self.contains(point) or \
                    str(point) in visited or \
                    len(res) > 10:
                return
            visited.add(str(point))
            if np.dot(norm, point) == intercept and \
                    (handler and handler(point)):
                # logger.debug(
                #     'Found feasible integer point %s on \n%s * x <= %s',
                #     point, norm, intercept
                # )
                res.append(point.copy())
            for d in directions:
                point += d
                dfs(point)
                point -= d

        tmp = initial_point.astype(np.int32)
        dfs(tmp)
        for d in directions:
            tmp += d
            if self.contains(tmp):
                dfs(tmp)
        #     if res is not None:
        #         return res
            tmp -= d
        dfs(tmp)

        sys.setrecursionlimit(998)
        return res

    def __str__(self):
        string = ''
        for vertex in self.vertices.values():
            string += str(vertex) + '\n'
        return string


class IntegralConvexHull(ConvexHull):
    def __init__(self, dimension, vertices):
        super().__init__(dimension)
        for v in vertices:
            self.add_vertex(v)
        self.integral_points = []
        with Timer() as t:
            self._iter_integral_points()
        logger.info('time of finding all integral points: %s', t.elapsed)

    def add_vertex(self, vertex):
        # all vertices are integral
        vertex.coordinate = vertex.coordinate.astype(np.int32)
        vertex.facets_norm = vertex.facets_norm.astype(np.int32)
        vertex.facets_intercept = vertex.facets_intercept.astype(np.int32)
        super().add_vertex(vertex)

    def _iter_integral_points(self):
        vertices_coordinate = np.zeros([len(self.vertices), self.dimension])
        for i, v in enumerate(self.vertices.values()):
            vertices_coordinate[i] = v.coordinate
        min_max_coordinate = np.stack([
            np.min(vertices_coordinate, axis=0),
            np.max(vertices_coordinate, axis=0)
        ]).astype(np.int32)
        coordinate_lists = [
            np.arange(min_max_coordinate[0, d], min_max_coordinate[1, d] + 1, dtype=np.int32)
            for d in range(self.dimension)
        ]
        self.integral_points = cartesian_product(*coordinate_lists)
        # for p in product(*coordinate_lists):
        #     if self.contains(p):
        #         self.integral_points.append(p)
        # self.integral_points = np.array(self.integral_points, dtype=np.int32)

    def add_facet(self, norm, intercept):
        # filter integral points
        self.integral_points = self.integral_points[
            np.where(np.dot(norm, self.integral_points.T) <= intercept)
        ]
        # filter vertex
        self.vertices = dict(filter(
            lambda v: np.dot(norm, v[1].coordinate) <= intercept,
            self.vertices.items()
        ))
        # logger.debug('filtered vertices: %s', self.vertices.keys())
        # logger.debug('find convex hull from %s', self.integral_points)
        with Timer() as t:
            new_convex_hull = scipy_convex_hull(self.integral_points)
        logger.info('elapsed time for quick-hull: %s', t.elapsed)
        added_vertices = []
        for v in new_convex_hull.vertices:
            vertex_coordinate = self.integral_points[v]
            norms = []
            intercepts = []
            if self.get_vertex(vertex_coordinate) is None:
                # facets_index = np.where(
                #     np.abs(np.dot(new_convex_hull.equations,
                #            np.append(vertex_coordinate, 1)) - 0) < 1e-7
                # )[0]
                facets_index = np.where(new_convex_hull.simplices == v)[0]
                for i in facets_index:
                    # integral_equation = get_integral_vec(
                    #     new_convex_hull.equations[i]
                    # )
                    # norms.append(integral_equation[:-1])
                    # intercepts.append(-integral_equation[-1])
                    new_norm, new_intercept = get_hyperplane(
                        [self.integral_points[v] for v in
                         new_convex_hull.simplices[i]]
                    )
                    new_norm = np.array(new_norm, dtype=np.int32)
                    new_intercept = np.array(new_intercept, dtype=np.int32)
                    # get the half-space
                    for v in new_convex_hull.vertices:
                        sign = np.dot(new_norm, self.integral_points[v])
                        if sign != new_intercept:
                            sign = 1 - 2 * int(sign > new_intercept)
                            new_norm *= sign
                            new_intercept *= sign
                            break
                    norms.append(new_norm)
                    intercepts.append(new_intercept)
                norms = np.array(norms, dtype=np.int32)
                intercepts = np.array(intercepts, dtype=np.int32)
                vertex = Vertex(vertex_coordinate, norms, intercepts)
                logger.debug('find new vertex %s', vertex)
                self.add_vertex(vertex)
                added_vertices.append(vertex)
        return added_vertices
