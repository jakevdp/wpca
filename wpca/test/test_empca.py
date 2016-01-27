import numpy as np
from numpy.testing import assert_allclose

from ..empca import orthonormalize, random_orthonormal, pca, empca

def norm_sign(X):
    i_max_abs = np.argmax(abs(X), 0)
    sgn = np.sign(X[i_max_abs, range(X.shape[1])])
    return X * sgn


def assert_columns_allclose_upto_sign(A, B, *args, **kwargs):
    assert_allclose(norm_sign(A), norm_sign(B), *args, **kwargs)


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


def test_empca_vs_pca():
    rand = np.random.RandomState(42)
    X = rand.randn(50, 5)
    W = np.ones_like(X)
    evecs1, coeff1 = empca(X, W, 5, niter=100, rseed=42)
    evecs2, coeff2 = pca(X, 5)

    assert_columns_allclose_upto_sign(evecs1.T, evecs2.T, rtol=1E-6)
    assert_columns_allclose_upto_sign(coeff1, coeff2, rtol=1E-6)
