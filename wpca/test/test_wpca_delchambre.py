import numpy as np
from numpy.testing import assert_allclose
from .. import pca, wpca_delchambre


def norm_sign(X):
    i_max_abs = np.argmax(abs(X), 0)
    sgn = np.sign(X[i_max_abs, range(X.shape[1])])
    return X * sgn


def assert_columns_allclose_upto_sign(A, B, *args, **kwargs):
    assert_allclose(norm_sign(A),
                    norm_sign(B), *args, **kwargs)


def test_wpca_delchambre_unweighted():
    rand = np.random.RandomState(0)
    X = rand.rand(5, 100)

    def check_results(ncomp):
        P1, C1, S1 = pca(X, ncomp)
        P2, C2, S2 = wpca_delchambre(X, ncomp)
        assert_allclose(S1, S2)
        assert_columns_allclose_upto_sign(P1, P2)
        assert_columns_allclose_upto_sign(C1.T, C2.T)

    for ncomp in range(1, 6):
        yield check_results, ncomp


def test_wpca_delchambre_outliers():
    rand = np.random.RandomState(0)
    X = rand.multivariate_normal([0, 0], [[12, 6],[6, 5]], size=1000).T
    ncomp = 2
    P1, C1, S1 = wpca_delchambre(X, ncomp)

    def check_results(n_outliers, noise_level, rtol):
        i = rand.randint(0, 2, size=n_outliers)
        j = rand.randint(0, 100, size=n_outliers)
        X2 = X.copy()
        X2[i, j] += noise_level * rand.randn(n_outliers)
        W2 = np.ones_like(X2)
        W2[i, j] = 1. / noise_level

        P2, C2, S2 = wpca_delchambre(X2, ncomp, W2)
        assert_columns_allclose_upto_sign(P1, P2, rtol=rtol)

    for (n_outliers, noise_level, rtol) in [(1, 20, 1E-3), (10, 20, 1E-2)]:
        yield check_results, n_outliers, noise_level, rtol
