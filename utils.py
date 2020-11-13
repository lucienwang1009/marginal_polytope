import numpy as np
import math

from logzero import logger
from functools import reduce
from scipy import linalg


def normalize_norm(norm):
    gcd = reduce(math.gcd, norm)
    if gcd:
        return np.array(norm / gcd, dtype=np.int)
    return norm


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
    norm = normalize_norm(norm)
    intercept = np.dot(norm, points[0])
    return norm, intercept
