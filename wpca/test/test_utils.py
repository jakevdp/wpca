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
    def check_random_orthonormal(n_samples, n_features):
        X = random_orthonormal(n_samples, n_features, 42)
        assert X.shape == (n_samples, n_features)
        assert_allclose(np.dot(X, X.T), np.eye(X.shape[0]), atol=1E-15)
    for n_samples in range(1, 6):
        yield check_random_orthonormal, n_samples, 5
