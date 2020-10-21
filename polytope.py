import numpy as np

from logzero import logger

from utils import coordinate2str


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
        self.coordinate = coordinate
        self.facets_norm = facets_norm
        self.facets_intercept = facets_intercept

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

    def __str__(self):
        string = ''
        for vertex in self.vertices.values():
            string += str(vertex) + '\n'
        return string
