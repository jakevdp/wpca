import numpy as np
from numpy.testing import assert_allclose, assert_raises
from wpca.tests.tools import assert_columns_allclose_upto_sign

from wpca import PCA, WPCA, EMPCA


ESTIMATORS = [WPCA, EMPCA]
KWDS = {WPCA: {}, EMPCA: {'random_state': 0}}


def test_weighted_mean():
    rand = np.random.RandomState(0)
    X = np.random.rand(30, 3)
    W = np.random.rand(30, 3)

    def test_weighted_mean(Estimator):
        pca = Estimator().fit(X, weights=W)
        assert_allclose(pca.mean_, np.average(X, weights=W, axis=0))

    for Estimator in ESTIMATORS:
        yield test_weighted_mean, Estimator


def test_fit_and_fit_transform():
    rand = np.random.RandomState(0)
    X = np.random.rand(30, 3)
    W = np.random.rand(30, 3)

    def check_results(Estimator, copy_data):
        if Estimator is EMPCA:
            # copy_data not yet implemented
            pca = Estimator(2, **KWDS[Estimator])
        else:
            pca = Estimator(2, copy_data=copy_data, **KWDS[Estimator])

        Y1 = pca.fit_transform(X.copy(), weights=W.copy())
        Y2 = pca.transform(X.copy(), weights=W.copy())

        assert_columns_allclose_upto_sign(Y1, Y2)

    for Estimator in ESTIMATORS:
        for copy_data in [True, False]:
            yield check_results, Estimator, copy_data



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


def test_bad_inputs():
    rand = np.random.RandomState(42)
    X = rand.rand(10, 3)
    W = np.ones_like(X)
    bad1 = (X > 0.8)
    bad2 = (X > 0.9)

    def check_mismatch(Estimator):
        pca = Estimator()
        assert_raises(ValueError, pca.fit, X, weights=W.T)

    def check_bad_inputs(Estimator, bad_val):
        X[bad1] = bad_val
        W[bad2] = 0
        pca = Estimator()
        assert_raises(ValueError, pca.fit, X, weights=W)

    for Estimator in ESTIMATORS:
        yield check_mismatch, Estimator
        for bad_val in [np.inf, np.nan]:
            yield check_bad_inputs, Estimator, bad_val


def test_copy_data():
    rand = np.random.RandomState(0)
    X = rand.multivariate_normal([0, 0], [[12, 6], [6, 5]], size=100)
    W = rand.rand(*X.shape)
    X_orig = X.copy()

    # with copy_data=True, X should not change
    pca1 = WPCA(copy_data=True)
    pca1.fit(X, weights=W)
    assert np.all(X == X_orig)

    # with copy_data=False, X should be overwritten
    pca2 = WPCA(copy_data=False)
    pca2.fit(X, weights=W)
    assert not np.allclose(X, X_orig)

    # all results should match
    assert_allclose(pca1.mean_, pca2.mean_)
    assert_allclose(pca1.components_, pca2.components_)
    assert_allclose(pca1.explained_variance_, pca2.explained_variance_)
