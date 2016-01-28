import numpy as np
from numpy.testing import assert_allclose

from .. import WPCA, EMPCA


ESTIMATORS = [WPCA, EMPCA]


def norm_sign(X):
    i_max_abs = np.argmax(abs(X), 0)
    sgn = np.sign(X[i_max_abs, range(X.shape[1])])
    return X * sgn


def assert_columns_allclose_upto_sign(A, B, *args, **kwargs):
    assert_allclose(norm_sign(A), norm_sign(B), *args, **kwargs)


def test_weighted_vs_unweighted():
    rand = np.random.RandomState(0)
    X = rand.multivariate_normal([0, 0], [[12, 6],[6, 5]], size=100)
    W = np.ones_like(X)

    def check_results(Estimator):
        pca1 = Estimator(2).fit(X)
        pca2 = Estimator(2).fit(X, W)
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

    for Estimator in ESTIMATORS:
        yield check_results, Estimator
