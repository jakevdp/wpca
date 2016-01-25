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

    shapes = [(10, 5), (6, 10)]
    data = {shape: rand.randn(*shape) for shape in shapes}

    def check_transform(Estimator, n_components, shape):
        X = data[shape]

        pca = Estimator(n_components).fit(X)
        skpca = SKPCA(n_components).fit(X)
        assert_columns_allclose_upto_sign(pca.components_.T,
                                          skpca.components_.T)
        assert_allclose(pca.explained_variance_,
                        skpca.explained_variance_)
        assert_allclose(pca.explained_variance_ratio_,
                        skpca.explained_variance_ratio_)

        Y = pca.transform(X)
        Ysk = skpca.transform(X)
        assert_columns_allclose_upto_sign(Y, Ysk)
        assert_allclose(pca.inverse_transform(Y),
                        skpca.inverse_transform(Ysk))

    for shape in shapes:
        for Estimator in (PCA, WPCA):
            for n_components in range(1, 6):
                yield check_transform, Estimator, n_components, shape


def test_transform():
    rand = np.random.RandomState(42)

    shapes = [(10, 5), (6, 10)]
    data = {shape: rand.randn(*shape) for shape in shapes}

    def check_transform(Estimator, n_components, shape):
        X = data[shape]
        pca = Estimator(n_components)
        Y1 = pca.fit_transform(X)
        Y2 = pca.transform(X)
        assert_allclose(Y1, Y2, atol=1E-14)

    # 10 points in 5 dimensions
    for shape in shapes:
        for Estimator in (PCA, WPCA):
            for n in range(1, 6):
                yield check_transform, Estimator, n, shape


def test_inverse_transform():
    rand = np.random.RandomState(42)

    shapes = [(10, 5), (6, 10)]
    data = {shape: rand.randn(*shape) for shape in shapes}

    def check_inverse_transform(Estimator, n_components, shape):
        X = rand.randn(*shape)
        pca = Estimator(n_components)
        Y = pca.fit_transform(X)
        assert_allclose(X, pca.inverse_transform(Y))

    for shape in shapes:
        for Estimator in (PCA, WPCA):
            yield check_inverse_transform, Estimator, min(shape), shape


def test_with_outliers():
    rand = np.random.RandomState(0)
    X = rand.multivariate_normal([0, 0], [[12, 6],[6, 5]], size=1000)
    pca = WPCA(2).fit(X)

    def check_results(n_outliers, noise_level, rtol):
        i = rand.randint(0, 100, size=n_outliers)
        j = rand.randint(0, 2, size=n_outliers)
        X2 = X.copy()
        X2[i, j] += noise_level * rand.randn(n_outliers)
        W2 = np.ones_like(X2)
        W2[i, j] = 1. / noise_level

        pca2 = WPCA(2).fit(X2, W2)
        assert_columns_allclose_upto_sign(pca.components_.T,
                                          pca2.components_.T,
                                          rtol=rtol)

    for (n_outliers, noise_level, rtol) in [(1, 20, 1E-3), (10, 20, 1E-2)]:
        yield check_results, n_outliers, noise_level, rtol


def test_with_nans():
    rand = np.random.RandomState(0)
    X = rand.rand(100, 10)

    i = rand.randint(0, 100, size=100)
    j = rand.randint(0, 2, size=100)
    X[i, j] = 0
    W = np.ones_like(X)
    W[i, j] = 0

    X2 = X.copy()
    X2[i, j] = np.nan

    pca1 = WPCA(2).fit(X, W)
    pca2 = WPCA(2).fit(X2, W)
    assert_columns_allclose_upto_sign(pca1.components_.T, pca2.components_.T)

    Y1 = pca1.transform(X, W)
    Y2 = pca2.transform(X2, W)
    assert_columns_allclose_upto_sign(Y1, Y2)

    Z1 = pca1.reconstruct(X, W)
    Z2 = pca2.reconstruct(X2, W)
    assert_allclose(Z1, Z2)
