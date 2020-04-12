# expm

> Trying to improve performance on Matrix Exponentials in Python

This project exists purely out of necessity. The 'truth benchmark' I will use for all testing here will be the SciPy implementation of matrix exponentials
([scipy.linalg.expm](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.expm.html)).

The first step in trying to make performant code is to set strict bounds on the problem you are willing to solve. So, the restrictions I will put on my implentation of `expm` will be:

* The code must be callable from Python v3.
* The matrix will be real-valued.
* The performance will not be tested on matricies of size < 10x10.
* The performance will not be tested on matricies of size > 100x100.

