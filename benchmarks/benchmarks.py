""" These are the benchmarks against which we will test various versions of matrix exponentials in Python.
"""
# general imports
import numpy as np
import scipy.sparse as sparse
from time import time
# importing various matrix exponential implementations
from expm import expm as expm_goal
from pure_python1 import expm as expm_python1
from scipy.linalg import expm as expm_scipy


def main():
    test_matrices = []

    for n in range(15, 40, 1):
        for d in [x / 100 for x in range(10, 101)]:
            test_matrices.append(sparse.random(n, n, density=d).toarray())

    for n in range(10, 101, 10):
        for d in [0.25, 0.5, 0.9]:
            test_matrices.append(sparse.random(n, n, density=d).toarray())

    start = time()
    for a in test_matrices:
        _ = expm_scipy(a)
    end = time()

    print('SciPy impementation:\t{0} seconds'.format(round(end - start, 5)))

    start = time()
    for a in test_matrices:
        _ = expm_python1(a)
    end = time()

    print('A purely Python impementation:\t{0} seconds'.format(round(end - start, 5)))

    start = time()
    for a in test_matrices:
        _ = expm_goal(a)
    end = time()

    print('The goal impementation:\t{0} seconds'.format(round(end - start, 5)))


if __name__ == '__main__':
    main()
