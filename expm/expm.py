import numpy as np
from numpy.linalg import norm, solve
from numpy import eye, dot, ceil, log2


def expm(A):
    """ Compute the matrix exponential using Pade approximation
    https://github.com/rngantner/Pade_PyCpp/blob/master/src/expm.py

    Args:
        A: Matrix (shape(M,M)) to be exponentiated
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
    Q = -U + V  # q_m(A) : denominator
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
    b = tuple([120., 60., 12., 1.])
    n = A.shape[0]
    ident = eye(n, n, dtype='float64')
    A2 = dot(A, A)
    U = dot(A, (b[3] * A2 + b[1] * ident))
    V = b[2] * A2 + b[0] * ident
    return U, V


def _pade5(A):
    """ Helper method for expm

    Args:
        A (np.array): input Matrix
    Returns:
        tuple: Mystery Components
    """
    b = tuple([30240., 15120., 3360., 420., 30., 1.])
    n = A.shape[0]
    ident = eye(n, n, dtype='float64')
    A2 = dot(A, A)
    A4 = dot(A2, A2)
    U = dot(A, b[5] * A4 + b[3] * A2 + b[1] * ident)
    V = b[4] * A4 + b[2] * A2 + b[0] * ident
    return U, V


def _pade7(A):
    """ Helper method for expm

    Args:
        A (np.array): input Matrix
    Returns:
        tuple: Mystery Components
    """
    b = tuple([17297280., 8648640., 1995840., 277200., 25200., 1512., 56., 1.])
    n = A.shape[0]
    ident = eye(n, n, dtype='float64')
    A2 = dot(A, A)
    A4 = dot(A2, A2)
    A6 = dot(A4, A2)
    U = dot(A, b[7] * A6 + b[5] * A4 + b[3] * A2 + b[1] * ident)
    V = b[6] * A6 + b[4] * A4 + b[2] * A2 + b[0] * ident
    return U, V


def _pade9(A):
    """ Helper method for expm

    Args:
        A (np.array): input Matrix
    Returns:
        tuple: Mystery Components
    """
    b = tuple([17643225600., 8821612800., 2075673600., 302702400., 30270240., 2162160., 110880., 3960., 90., 1.])
    n = A.shape[0]
    ident = eye(n, n, dtype='float64')
    A2 = dot(A, A)
    A4 = dot(A2, A2)
    A6 = dot(A4, A2)
    A8 = dot(A6, A2)
    U = dot(A, b[9] * A8 + b[7] * A6 + b[5] * A4 + b[3] * A2 + b[1] * ident)
    V = b[8] * A8 + b[6] * A6 + b[4] * A4 + b[2] * A2 + b[0] * ident
    return U, V


def _pade13(A):
    """ Helper method for expm

    Args:
        A (np.array): input Matrix
    Returns:
        tuple: Mystery Components
    """
    b = tuple([64764752532480000., 32382376266240000., 7771770303897600., 1187353796428800., 129060195264000.,
               10559470521600., 670442572800., 33522128640., 1323241920., 40840800., 960960., 16380., 182., 1.])
    n = A.shape[0]
    ident = eye(n, n, dtype='float64')
    A2 = dot(A, A)
    A4 = dot(A2, A2)
    A6 = dot(A4, A2)
    U = dot(A, dot(A6, b[13] * A6 + b[11] * A4 + b[9] * A2) + b[7] * A6 + b[5] * A4 + b[3] * A2 + b[1] * ident)
    V = dot(A6, b[12] * A6 + b[10] * A4 + b[8] * A2) + b[6] * A6 + b[4] * A4 + b[2] * A2 + b[0] * ident
    return U, V
