from .. import PCA, WPCA
from sklearn.decomposition import PCA as SKPCA
import numpy as np
from numpy.testing import assert_allclose


def norm_sign(X):
    i_max_abs = np.argmax(abs(X), 0)
    sgn = np.sign(X[i_max_abs, range(X.shape[1])])
    return X * sgn


def assert_columns_allclose_upto_sign(A, B, *args, **kwargs):
    assert_allclose(norm_sign(A), norm_sign(B), *args, **kwargs)


def test_vs_sklearn():
    rand = np.random.RandomState(42)

    def check_vs_sklearn(Estimator, n_components, shape):
        X = rand.randn(*shape)

        pca = Estimator(n_components).fit(X)
        skpca = SKPCA(n_components).fit(X)

        Y = pca.transform(X)
        Ysk = skpca.transform(X)

        assert_columns_allclose_upto_sign(Y, Ysk)
        assert_columns_allclose_upto_sign(pca.components_.T,
                                          skpca.components_.T)

    # 10 points in 5 dimensions
    for shape in [(10, 5), (5, 10)]:
        for Estimator in (PCA, WPCA):
            for n in range(1, 6):
                yield check_vs_sklearn, Estimator, n, shape


def test_pca_transform():
    rand = np.random.RandomState(42)

    def check_transform(Estimator, n_components, shape):
        X = rand.randn(*shape)
        pca = Estimator(n_components)
        Y1 = pca.fit_transform(X)
        Y2 = pca.transform(X)
        assert_allclose(Y1, Y2, atol=1E-14)

    # 10 points in 5 dimensions
    for shape in [(10, 5), (5, 10)]:
        for Estimator in (PCA, WPCA):
            for n in range(1, 6):
                yield check_transform, Estimator, n, shape


def test_pca_inverse_transform():
    rand = np.random.RandomState(42)

    def check_inverse_transform(Estimator, n_components, shape):
        X = rand.randn(*shape)
        pca = Estimator(n_components)
        Y = pca.fit_transform(X)
        assert_allclose(X, pca.inverse_transform(Y))

    for shape in [(10, 5), (5, 10)]:
        for Estimator in (PCA, WPCA):
            yield check_inverse_transform, Estimator, min(shape), shape
