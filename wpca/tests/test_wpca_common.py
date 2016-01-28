import numpy as np
from numpy.testing import assert_allclose
from .tools import assert_columns_allclose_upto_sign

from .. import PCA, WPCA, EMPCA


ESTIMATORS = [WPCA, EMPCA]
KWDS = {WPCA: {}, EMPCA: {'random_state': 0}}


def test_constant_weights():
    rand = np.random.RandomState(0)
    X = rand.multivariate_normal([0, 0], [[12, 6], [6, 5]], size=100)
    W = np.ones_like(X)

    def check_results(Estimator):
        pca1 = Estimator(2, **KWDS[Estimator]).fit(X)
        pca2 = Estimator(2, **KWDS[Estimator]).fit(X, weights=W)
        assert_columns_allclose_upto_sign(pca1.components_.T,
                                          pca2.components_.T)
        assert_allclose(pca1.explained_variance_,
                        pca2.explained_variance_)
        assert_allclose(pca1.explained_variance_ratio_,
                        pca2.explained_variance_ratio_)

        Y1 = pca1.transform(X)
        Y2 = pca2.transform(X, W)
        assert_columns_allclose_upto_sign(Y1, Y2)

        X1 = pca1.inverse_transform(Y1)
        X2 = pca2.inverse_transform(Y2)
        assert_allclose(X1, X2)

        assert_columns_allclose_upto_sign(pca1.fit_transform(X),
                                          pca2.fit_transform(X, weights=W))

        assert_allclose(pca1.reconstruct(X),
                        pca2.reconstruct(X, W))

        assert_allclose(pca1.fit_reconstruct(X),
                        pca2.fit_reconstruct(X, weights=W))

    for Estimator in ESTIMATORS:
        yield check_results, Estimator


def test_outlier_weights():
    rand = np.random.RandomState(0)
    X = rand.multivariate_normal([0, 0], [[12, 6], [6, 5]], size=1000)
    pca = PCA(2).fit(X)

    def check_results(Estimator, n_outliers, noise_level, rtol):
        i = rand.randint(0, 100, size=n_outliers)
        j = rand.randint(0, 2, size=n_outliers)
        X2 = X.copy()
        X2[i, j] += noise_level * rand.randn(n_outliers)
        W2 = np.ones_like(X2)
        W2[i, j] = 1. / noise_level

        pca2 = Estimator(2, **KWDS[Estimator]).fit(X2, weights=W2)
        assert_columns_allclose_upto_sign(pca.components_.T,
                                          pca2.components_.T,
                                          rtol=rtol)

    for (n_outliers, noise_level, rtol) in [(1, 20, 1E-3), (10, 10, 3E-2)]:
        for Estimator in ESTIMATORS:
            yield check_results, Estimator, n_outliers, noise_level, rtol


def test_nan_weights():
    rand = np.random.RandomState(0)
    X = rand.rand(100, 10)

    i = rand.randint(0, 100, size=100)
    j = rand.randint(0, 2, size=100)
    X[i, j] = 0
    W = np.ones_like(X)
    W[i, j] = 0

    X2 = X.copy()
    X2[i, j] = np.nan

    def check_results(Estimator):
        pca1 = Estimator(2, **KWDS[Estimator]).fit(X, weights=W)
        pca2 = Estimator(2, **KWDS[Estimator]).fit(X2, weights=W)
        assert_columns_allclose_upto_sign(pca1.components_.T,
                                          pca2.components_.T)
        assert_allclose(pca1.explained_variance_,
                        pca2.explained_variance_)
        assert_allclose(pca1.explained_variance_ratio_,
                        pca2.explained_variance_ratio_)

        Y1 = pca1.transform(X, W)
        Y2 = pca2.transform(X2, W)
        assert_columns_allclose_upto_sign(Y1, Y2)

        Z1 = pca1.reconstruct(X, W)
        Z2 = pca2.reconstruct(X2, W)
        assert_allclose(Z1, Z2)

    for Estimator in ESTIMATORS:
        yield check_results, Estimator
