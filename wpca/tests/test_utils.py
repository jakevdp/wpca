import numpy as np
from numpy.testing import assert_allclose

from ..utils import orthonormalize, random_orthonormal


def test_orthonormalize():
    rand = np.random.RandomState(42)
    X = rand.randn(3, 4)
    X2 = orthonormalize(X)
    assert_allclose(X[0] / np.linalg.norm(X[0]), X2[0])
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
