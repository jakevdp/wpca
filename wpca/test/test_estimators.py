from .. import PCA
import numpy as np
from numpy.testing import assert_allclose

def test_pca_transform():
    rand = np.random.RandomState(42)

    def check_transform(n_components, X):
        pca = PCA(n_components)
        Y1 = pca.fit_transform(X)
        Y2 = pca.transform(X)
        assert_allclose(Y1, Y2, atol=1E-14)

    # 10 points in 5 dimensions
    X = rand.randn(10, 5)
    for n in range(1, 6):
        yield check_transform, n, X

    # 5 points in 10 dimensions
    X = rand.randn(5, 10)
    for n in range(1, 6):
        yield check_transform, n, X



def test_pca_inverse_transform():
    rand = np.random.RandomState(42)

    # 10 points in 5 dimensions
    X = rand.randn(10, 5)
    pca = PCA(5)
    Y = pca.fit_transform(X)
    assert_allclose(X, pca.inverse_transform(Y))

    # 5 points in 10 dimensions
    X = rand.randn(10, 5)
    pca = PCA(5)
    Y = pca.fit_transform(X)
    assert_allclose(X, pca.inverse_transform(Y))
