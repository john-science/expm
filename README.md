# expm
[![Build Status](https://travis-ci.com/theJollySin/expm.svg?branch=master)](https://travis-ci.com/theJollySin/expm)
[![codecov](https://codecov.io/gh/theJollySin/expm/branch/master/graph/badge.svg)](https://codecov.io/gh/theJollySin/expm)

> Improving the performance of Matrix Exponentials in Python

This project exists purely out of necessity. The 'truth benchmark' I will use for all testing here will be the SciPy implementation of matrix exponentials
([scipy.linalg.expm](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.expm.html)).

I would like to point out that the fastest implementation I have found in preparation for this project was from [rngantner on GitHub](https://github.com/rngantner/Pade_PyCpp/blob/master/src/expm.py).

The first step in trying to make performant code is to set strict bounds on the problem you are willing to solve. So, the restrictions I will put on my implentation of `expm` will be:

* The code must be callable from Python v3.5 to v3.7.
* The performance testing will be done on Python v3.6.
* The matrix will be real-valued.
* The performance will only be tested against 2D arrays.
* The performance will not be tested on matricies of size < 10x10.
* The performance will not be tested on matricies of size > 100x100.

