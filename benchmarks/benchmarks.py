""" These are the benchmarks against which we will test various versions of matrix exponentials in Python.
"""
# general imports
import numpy as np
import scipy.sparse as sparse
from time import time
# importing various matrix exponential implementations
from expm import expm as expm_goal
from expm.expm_test import expm_test
from pure_python_vanilla import expm as expm_python1
from pure_python_one_function import expm as expm_python2
from scipy.linalg import expm as expm_scipy


def main():
    test_matrices = generate_test_matricies()

    print_time_results(expm_scipy, test_matrices, 'SciPy implementation')
    print_time_results(expm_python1, test_matrices, 'Pure Python - Vanilla implementation')
    print_time_results(expm_python2, test_matrices, 'Pure Python - One Function implementation')
    print_time_results(expm_goal, test_matrices, 'The goal implementation')
    print_time_results(expm_test, test_matrices, 'The pyx test implementation')


def generate_test_matricies():
    """ Generate some random, sparse matricies of various sizes for testing

    Returns:
        list: random matricies for testing
    """
    test_matrices = []

    for _ in range(10):
        for n in range(15, 40):
            for d in [x / 100 for x in range(10, 101)]:
                test_matrices.append(sparse.random(n, n, density=d).toarray())

    for n in range(10, 101):
        for d in [0.25, 0.5, 0.9]:
            test_matrices.append(sparse.random(n, n, density=d).toarray())

    return test_matrices


def print_time_results(exp_m, test_matrices, s):
    """ Run some version of matrix exponential against a list of arrays, and pretty print the timing results

    Args:
        exp_m (function): some variation of matrix exponentials
        test_matrices (list): 2D NumPy arrays for testing
        s (str): pretty strings to print out, describing which matrix exponential version this is
    Returns: None
    """
    start = time()
    for a in test_matrices:
        _ = exp_m(a)
    end = time()

    print('{0}:\t{1} seconds'.format(s, round(end - start, 3)))


if __name__ == '__main__':
    main()

