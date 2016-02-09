from itertools import chain, combinations

import numpy as np
from numpy.testing import assert_allclose

from wpca.tests.tools import assert_allclose_upto_sign
from wpca.utils import orthonormalize, random_orthonormal, weighted_mean


def test_orthonormalize():
    rand = np.random.RandomState(42)
    X = rand.randn(3, 4)
    X2 = orthonormalize(X)
    assert_allclose_upto_sign(X[0] / np.linalg.norm(X[0]), X2[0])
    assert_allclose(np.dot(X2, X2.T), np.eye(X2.shape[0]), atol=1E-15)


def test_random_orthonormal():
    def check_random_orthonormal(N, M, rows):
        X = random_orthonormal(N, M, rows=rows, random_state=42)
        assert X.shape == (N, M)
        if rows:
            C = np.dot(X, X.T)
        else:
            C = np.dot(X.T, X)
        assert_allclose(C, np.eye(C.shape[0]), atol=1E-15)
    for M in [5]:
        for N in range(1, M + 1):
            yield check_random_orthonormal, N, M, True
            yield check_random_orthonormal, M, N, False


def test_weighted_mean():
    def check_weighted_mean(shape, axis):
        rand = np.random.RandomState(0)
        x = rand.rand(*shape)
        w = rand.rand(*shape)
        wm = weighted_mean(x, w, axis)
        assert_allclose(wm, np.average(x, axis, w))
        assert_allclose(wm, (w * x).sum(axis) / w.sum(axis))


    for ndim in range(1, 5):
        shape = tuple(range(3, 3 + ndim))
        axis_tuples = chain(*(combinations(range(ndim), nax)
                            for nax in range(ndim + 1)))
        for axis in chain([None], range(ndim), axis_tuples):
            yield check_weighted_mean, shape, axis
