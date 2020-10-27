import numpy as np


def coordinate2str(coordinate):
    co_list = coordinate.tolist()
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
        res[i] = ((-1) ** i) * np.linalg.det(tmp)
    return res
