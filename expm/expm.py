""" Source of this EXPM implmentation: https://github.com/theJollySin/expm
"""
import numpy as np
from numpy.linalg import norm, solve
from numpy import eye, dot
import cython
if not cython.compiled:
    from numpy import ceil, log2


def expm(A):
    """ Compute the matrix exponential using Pade approximation
    https://github.com/rngantner/Pade_PyCpp/blob/master/src/expm.py

    Args:
        A (np.array): Matrix (shape(M,M)) to be exponentiated
    Returns:
        np.array: Matrix (shape(M,M)) exponential of A
    """
    A_L1 = norm(A, 1)
    n_squarings = 0

    if A_L1 < 1.495585217958292e-2:
        U, V = _pade3(A)
    elif A_L1 < 2.539398330063230e-1:
        U, V = _pade5(A)
    elif A_L1 < 9.504178996162932e-1:
        U, V = _pade7(A)
    elif A_L1 < 2.097847961257068:
        U, V = _pade9(A)
    else:
        n_squarings = max(0, int(ceil(log2(A_L1 / 5.371920351148152))))
        A = A / 2 ** n_squarings
        U, V = _pade13(A)

    P = U + V  # p_m(A) : numerator
    Q = V - U  # q_m(A) : denominator
    R = solve(Q, P)

    # squaring step to undo scaling
    for _ in range(n_squarings):
        R = dot(R, R)

    return R


def _pade3(A):
    """ Helper method for expm

    Args:
        A (np.array): input Matrix
    Returns:
        tuple: Mystery Components
    """
    n = len(A)
    ident = eye(n, n, dtype=np.float64)
    A2 = dot(A, A)
    U = dot(A, (pade3_b3 * A2 + pade3_b1 * ident))
    V = pade3_b2 * A2 + pade3_b0 * ident
    return U, V


def _pade5(A):
    """ Helper method for expm

    Args:
        A (np.array): input Matrix
    Returns:
        tuple: Mystery Components
    """
    n = len(A)
    ident = eye(n, n, dtype=np.float64)
    A2 = dot(A, A)
    A4 = dot(A2, A2)
    U = dot(A, pade5_b5 * A4 + pade5_b3 * A2 + pade5_b1 * ident)
    V = pade5_b4 * A4 + pade5_b2 * A2 + pade5_b0 * ident
    return U, V


def _pade7(A):
    """ Helper method for expm

    Args:
        A (np.array): input Matrix
    Returns:
        tuple: Mystery Components
    """
    n = len(A)
    ident = eye(n, n, dtype=np.float64)
    A2 = dot(A, A)
    A4 = dot(A2, A2)
    A6 = dot(A4, A2)
    U = dot(A, pade7_b7 * A6 + pade7_b5 * A4 + pade7_b3 * A2 + pade7_b1 * ident)
    V = pade7_b6 * A6 + pade7_b4 * A4 + pade7_b2 * A2 + pade7_b0 * ident
    return U, V


def _pade9(A):
    """ Helper method for expm

    Args:
        A (np.array): input Matrix
    Returns:
        tuple: Mystery Components
    """
    n = len(A)
    ident = eye(n, n, dtype=np.float64)
    A2 = dot(A, A)
    A4 = dot(A2, A2)
    A6 = dot(A4, A2)
    A8 = dot(A6, A2)
    U = dot(A, pade9_b9 * A8 + pade9_b7 * A6 + pade9_b5 * A4 + pade9_b3 * A2 + pade9_b1 * ident)
    V = pade9_b8 * A8 + pade9_b6 * A6 + pade9_b4 * A4 + pade9_b2 * A2 + pade9_b0 * ident
    return U, V


def _pade13(A):
    """ Helper method for expm

    Args:
        A (np.array): input Matrix
    Returns:
        tuple: Mystery Components
    """
    n = len(A)
    ident = eye(n, n, dtype=np.float64)
    A2 = dot(A, A)
    A4 = dot(A2, A2)
    A6 = dot(A4, A2)
    U = dot(A, dot(A6, pade13_b13 * A6 + pade13_b11 * A4 + pade13_b9 * A2) + pade13_b7 * A6 + pade13_b5 * A4 + pade13_b3 * A2 + pade13_b1 * ident)
    V = dot(A6, pade13_b12 * A6 + pade13_b10 * A4 + pade13_b8 * A2) + pade13_b6 * A6 + pade13_b4 * A4 + pade13_b2 * A2 + pade13_b0 * ident
    return U, V


# So many constants
pade3_b0 = 120.
pade3_b1 = 60.
pade3_b2 = 12.
pade3_b3 = 1.

pade5_b0 = 30240.
pade5_b1 = 15120.
pade5_b2 = 3360.
pade5_b3 = 420.
pade5_b4 = 30.
pade5_b5 = 1.

pade7_b0 = 17297280.
pade7_b1 = 8648640.
pade7_b2 = 1995840.
pade7_b3 = 277200.
pade7_b4 = 25200.
pade7_b5 = 1512.
pade7_b6 = 56.
pade7_b7 = 1.

pade9_b0 = 17643225600.
pade9_b1 = 8821612800.
pade9_b2 = 2075673600.
pade9_b3 = 302702400.
pade9_b4 = 30270240.
pade9_b5 = 2162160.
pade9_b6 = 110880.
pade9_b7 = 3960.
pade9_b8 = 90.
pade9_b9 = 1.

pade13_b0 = 64764752532480000.
pade13_b1 = 32382376266240000.
pade13_b2 = 7771770303897600.
pade13_b3 = 1187353796428800.
pade13_b4 = 129060195264000.
pade13_b5 = 10559470521600.
pade13_b6 = 670442572800.
pade13_b7 = 33522128640.
pade13_b8 = 1323241920.
pade13_b9 = 40840800.
pade13_b10 = 960960.
pade13_b11 = 16380.
pade13_b12 = 182.
pade13_b13 = 1.
