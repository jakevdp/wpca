import numpy as np
from numpy.testing import assert_allclose
from .. import pca, wpca_delchambre


def norm_sign(X):
    # TODO: this may not work when column sum is zero
    sgn = np.sign(X.sum(0))
    sgn[sgn == 0] = 1
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
