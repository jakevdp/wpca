# Weighted Principal Component Analysis in Python

*Author: Jake VanderPlas*

[![version status](http://img.shields.io/pypi/v/wpca.svg?style=flat)](https://pypi.python.org/pypi/wpca)
[![downloads](http://img.shields.io/pypi/dm/wpca.svg?style=flat)](https://pypi.python.org/pypi/wpca)
[![build status](http://img.shields.io/travis/jakevdp/wpca/master.svg?style=flat)](https://travis-ci.org/jakevdp/wpca)
[![license](http://img.shields.io/badge/license-BSD-blue.svg?style=flat)](https://github.com/jakevdp/wpca/blob/master/LICENSE)

This repository contains several implementations of Weighted Principal Component
Analysis, using a very similar interface to scikit-learn's
[``sklearn.decomposition.PCA``](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html):

- ``wpca.WPCA`` uses a direct decomposition of a weighted covariance matrix to
  compute principal vectors, and then a weighted least squares optimization
  to compute principal components. It is based on the algorithm presented
  in [Delchambre (2014)](http://arxiv.org/abs/1412.4533)

- ``wpca.EMPCA`` uses an iterative expectation-maximization approach to solve
  simultaneously for the principal vectors and principal components of
  weighted data. It is based on the algorithm presented in
  [Bailey (2012)](http://arxiv.org/abs/1208.4122).

- ``wpca.PCA`` is a standard non-weighted PCA implemented using the singular
  value decomposition. It is mainly included for the sake of testing.

## Examples and Documentation

For an example application of a weighted PCA approach, See
[WPCA-Example.ipynb](WPCA-Example.ipynb).

## Installation & Dependencies

This package has the following requirements:

- Python versions 2.7, or 3.4+
- [numpy](http://numpy.org/) (tested with version 1.10)
- [scipy](http://scipy.org/) (tested with version 0.16)
- [scikit-learn](http://scikit-learn.org/) (tested with version 0.17)
- [nose](http://nose.readthedocs.org/) (optional) to run unit tests.

With these requirements satisfied, you can install this package by running
```
pip install wpca
```
or to install from the source tree, run
```
$ python setup.py install
```

To run the suite of unit tests, make sure ``nose`` is installed and run
```
$ nosetests wpca
```
