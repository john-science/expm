""" These are the benchmarks against which we will test various versions of matrix exponentials in Python.
"""
# general imports
import pyximport;
pyximport.install()
import numpy as np
import scipy.sparse as sparse
from time import time
# importing various matrix exponential implementations
from expm import expm as expm_goal
from expm.expm_test import expm_test
from pure_python1 import expm as expm_python1
from scipy.linalg import expm as expm_scipy


def main():
    test_matrices = generate_test_matricies()

    print_time_results(expm_scipy, 'SciPy implementation', test_matrices)
    print_time_results(expm_python1, 'Pure Python implementation', test_matrices)
    print_time_results(expm_goal, 'The goal implementation', test_matrices)
    print_time_results(expm_test, 'The pyx test implementation', test_matrices)


def generate_test_matricies():
    """ TODO """
    test_matrices = []

    for _ in range(10):
        for n in range(15, 40, 1):
            for d in [x / 100 for x in range(10, 101)]:
                test_matrices.append(sparse.random(n, n, density=d).toarray())

    for n in range(10, 101):
        for d in [0.25, 0.5, 0.9]:
            test_matrices.append(sparse.random(n, n, density=d).toarray())

    return test_matrices


def print_time_results(exp_m, str, test_matrices):
    """ TODO """
    start = time()
    for a in test_matrices:
        _ = exp_m(a)
    end = time()

    print(str + ':\t{0} seconds'.format(round(end - start, 5)))


if __name__ == '__main__':
    main()
