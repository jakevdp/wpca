from .. import PCA, WPCA, EMPCA
from sklearn.decomposition import PCA as SKPCA
import numpy as np
from numpy.testing import assert_allclose
from .tools import assert_columns_allclose_upto_sign


ESTIMATORS = [PCA, WPCA, EMPCA]
KWDS = {PCA:{}, WPCA:{}, EMPCA:{'random_state':0}}
SHAPES = [(10, 5), (6, 10)]
rand = np.random.RandomState(42)
DATA = {shape: rand.randn(*shape) for shape in SHAPES}


def test_components_vs_sklearn():
    def check_components(Estimator, n_components, shape):
        X = DATA[shape]

        pca = Estimator(n_components, **KWDS[Estimator]).fit(X)
        skpca = SKPCA(n_components).fit(X)

        assert_columns_allclose_upto_sign(pca.components_.T,
                                          skpca.components_.T)

    for Estimator in ESTIMATORS:
        for shape in SHAPES:
            for n_components in range(1, 6):
                yield check_components, Estimator, n_components, shape


def test_explained_variance_vs_sklearn():
    def check_explained_variance(Estimator, n_components, shape):
        X = DATA[shape]

        pca = Estimator(n_components, **KWDS[Estimator]).fit(X)
        skpca = SKPCA(n_components).fit(X)

        assert_allclose(pca.explained_variance_,
                        skpca.explained_variance_)
        assert_allclose(pca.explained_variance_ratio_,
                        skpca.explained_variance_ratio_)

    for Estimator in ESTIMATORS:
        for shape in SHAPES:
            for n_components in range(1, 6):
                yield check_explained_variance, Estimator, n_components, shape


def test_transform_vs_sklearn():
    def check_transform(Estimator, n_components, shape):
        X = DATA[shape]

        pca = Estimator(n_components, **KWDS[Estimator])
        skpca = SKPCA(n_components)

        Y = pca.fit_transform(X)
        Ysk = skpca.fit_transform(X)
        assert_columns_allclose_upto_sign(Y, Ysk)

    for Estimator in ESTIMATORS:
        for shape in SHAPES:
            for n_components in range(1, 6):
                yield check_transform, Estimator, n_components, shape


def test_transform_vs_fit_transform():
    def check_transform(Estimator, n_components, shape):
        X = DATA[shape]

        Y1 = Estimator(n_components, **KWDS[Estimator]).fit(X).transform(X)
        Y2 = Estimator(n_components, **KWDS[Estimator]).fit_transform(X)

        assert_columns_allclose_upto_sign(Y1, Y2)

    for Estimator in ESTIMATORS:
        for shape in SHAPES:
            for n_components in range(1, 6):
                yield check_transform, Estimator, n_components, shape


def test_pca_reconstruct():
    def check_reconstruct(Estimator, shape):
        X = DATA[shape]

        pca = Estimator(n_components=min(shape), **KWDS[Estimator])

        assert_allclose(X, pca.fit(X).reconstruct(X))
        assert_allclose(X, pca.fit_reconstruct(X))

    for Estimator in ESTIMATORS:
        for shape in SHAPES:
            yield check_reconstruct, Estimator, shape
