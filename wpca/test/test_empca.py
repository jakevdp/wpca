import numpy as np
from numpy.testing import assert_allclose

from ..empca import pca, empca

def norm_sign(X):
    i_max_abs = np.argmax(abs(X), 0)
    sgn = np.sign(X[i_max_abs, range(X.shape[1])])
    return X * sgn


def assert_columns_allclose_upto_sign(A, B, *args, **kwargs):
    assert_allclose(norm_sign(A), norm_sign(B), *args, **kwargs)


def test_empca_vs_pca():
    rand = np.random.RandomState(42)
    X = rand.randn(50, 5)
    W = np.ones_like(X)
    evecs1, coeff1 = empca(X, W, 5, niter=100, random_state=42)
    evecs2, coeff2 = pca(X, 5)

    assert_columns_allclose_upto_sign(evecs1.T, evecs2.T, rtol=1E-6)
    assert_columns_allclose_upto_sign(coeff1, coeff2, rtol=1E-6)
