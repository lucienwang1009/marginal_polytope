import numpy as np
import math

from logzero import logger
from functools import reduce
from scipy import linalg
from fractions import Fraction


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
    orthogonal_vec = generalized_cross_product(vecs).astype(np.int32)
    return orthogonal_vec


def get_hyperplane(points):
    points = np.array(points)
    norm = get_orthogonal_vector(points)
    if norm[0] < 0:
        norm = -norm
    norm = normalize_norm(norm)
    intercept = np.dot(norm, points[0])
    return norm, intercept
