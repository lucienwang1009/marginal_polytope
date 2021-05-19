import numpy as np
import math
import mpl_toolkits.mplot3d as a3

from logzero import logger
from functools import reduce
from scipy import linalg
from scipy.spatial import ConvexHull
from scipy.spatial import HalfspaceIntersection
from fractions import Fraction
from matplotlib import pyplot as plt


class Faces():
    def __init__(self, tri, sig_dig=12, method="convexhull"):
        self.method = method
        self.tri = np.around(np.array(tri), sig_dig)
        self.grpinx = list(range(len(tri)))
        norms = np.around([self.norm(s) for s in self.tri], sig_dig)
        _, self.inv = np.unique(norms, return_inverse=True, axis=0)

    def norm(self, sq):
        cr = np.cross(sq[2] - sq[0], sq[1] - sq[0])
        return np.abs(cr / np.linalg.norm(cr))

    def isneighbor(self, tr1, tr2):
        a = np.concatenate((tr1, tr2), axis=0)
        return len(a) == len(np.unique(a, axis=0)) + 2

    def order(self, v):
        if len(v) <= 3:
            return v
        v = np.unique(v, axis=0)
        n = self.norm(v[:3])
        y = np.cross(n, v[1] - v[0])
        y = y / np.linalg.norm(y)
        c = np.dot(v, np.c_[v[1] - v[0], y])
        if self.method == "convexhull":
            h = ConvexHull(c)
            return v[h.vertices]
        else:
            mean = np.mean(c, axis=0)
            d = c - mean
            s = np.arctan2(d[:, 0], d[:, 1])
            return v[np.argsort(s)]

    def simplify(self):
        for i, tri1 in enumerate(self.tri):
            for j, tri2 in enumerate(self.tri):
                if j > i:
                    if self.isneighbor(tri1, tri2) and \
                            self.inv[i] == self.inv[j]:
                        self.grpinx[j] = self.grpinx[i]
        groups = []
        for i in np.unique(self.grpinx):
            u = self.tri[self.grpinx == i]
            u = np.concatenate([d for d in u])
            u = self.order(u)
            groups.append(u)
        return groups


def gcd_vec(vec):
    return reduce(math.gcd, vec)


def normalize_norm(norm):
    gcd = gcd_vec(norm)
    if gcd:
        return [int(num / gcd) for num in norm]
    return list(norm)


def get_integral_vec(vec):
    fractions = [Fraction('{:.7f}'.format(i)) for i in vec]
    denominator_product = reduce(lambda x, y: x * y,
                                 [f.denominator for f in fractions])
    integral_vec = []
    for i, f in enumerate(fractions):
        integral_vec.append(
            int(f.numerator * denominator_product / f.denominator)
        )
    return normalize_norm(integral_vec)


def coordinate2str(coordinate):
    co_list = list(coordinate)
    for i, l in enumerate(co_list):
        # round to integer for numeric problem
        if abs(round(l) - l) < 1e-7:
            co_list[i] = round(l)
    return str(co_list)


def generalized_cross_product(A):
    dimension = A.shape[1]
    res = np.zeros(dimension)
    for i in range(dimension):
        tmp = np.hstack(
            (A[:, :i], A[:, i + 1:])
        )
        res[i] = ((-1) ** i) * linalg.det(tmp)
    return res


def get_orthogonal_vector(points):
    vecs = []
    for p in points[1:]:
        vecs.append(p - points[0])
    vecs = np.array(vecs)
    orthogonal_vec = generalized_cross_product(vecs)
    return orthogonal_vec


def get_hyperplane(points):
    # points must be integral
    assert isinstance(points[0][0], (int, np.integer))
    points = np.array(points)
    norm = get_orthogonal_vector(points)
    if norm[0] < 0:
        norm = -norm
    # round to integral norm
    norm = [int(round(n)) for n in norm]
    norm = normalize_norm(norm)
    intercept = np.dot(norm, points[0])
    return norm, intercept


def inverse_dft_with_ln_input(x):
    pass


def plot_convex_hull(convex_hull, file_name=None):
    dimension = convex_hull.points.shape[1]
    if dimension > 3:
        logger.warning('Cannot show convex hull in 4D space')
        return

    if dimension == 3:
        ax = a3.Axes3D(plt.figure())
        org_triangles = [convex_hull.points[s] for s in convex_hull.simplices]
        f = Faces(org_triangles)
        g = f.simplify()

        colors = list(map("C{}".format, range(len(g))))
        pc = a3.art3d.Poly3DCollection(g, facecolor=colors, edgecolor='k', alpha=0.9)
        ax.add_collection3d(pc)
        ax.dist = 10
        ax.azim = 30
        ax.elev = 10
        ax.set_xlim([convex_hull.min_bound[0], convex_hull.max_bound[0]])
        ax.set_ylim([convex_hull.min_bound[1], convex_hull.max_bound[1]])
        ax.set_zlim([convex_hull.min_bound[2], convex_hull.max_bound[2]])
    elif dimension == 2:
        for s in convex_hull.simplices:
            plt.plot(*convex_hull.points[s, :].T, 'r-', lw=2)
        plt.plot(convex_hull.points[convex_hull.vertices, 0], convex_hull.points[convex_hull.vertices, 1], 'ro')

    if file_name is None:
        plt.show()
    else:
        plt.savefig(file_name)


def cartesian_product(*arrays):
    """
    https://stackoverflow.com/questions/11144513/cartesian-product-of-x-and-y-array-points-into-single-array-of-2d-points
    """

    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([la] + [len(a) for a in arrays], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[i, ...] = a
    return arr.reshape(la, -1).T


def herbrand_size(mln):
    domains = mln.domains
    preds = mln.predicates
    total = 0
    for p in preds:
        tmp = 1
        for d in p.argdoms:
            tmp *= len(domains[d])
        total += tmp
    return total


def world_size(mln):
    return 2 ** herbrand_size(mln)
