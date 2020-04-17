# cython: language_level=3
cimport cython
import numpy as np
cimport numpy as np
np.import_array()

# let's pull as many math functions as we can from C
# TODO: The types here are double, not np.whatever. Is that a problem?
cdef extern from "<math.h>" nogil:
    double ceil(double x)
    double log2(double x)

# trying to sort out these NumPy types
DTYPE = np.float
ctypedef np.float_t DTYPE_t


def expm_test(np.ndarray[DTYPE_t, ndim=2] A not None):
    """ Compute the matrix exponential using Pade approximation
    https://github.com/rngantner/Pade_PyCpp/blob/master/src/expm.py

    Args:
        A: Matrix (shape(M,M)) to be exponentiated
    Returns:
        np.array: Matrix (shape(M,M)) exponential of A
    """
    cdef int n = A.shape[0]
    cdef int n_squarings = 0
    cdef np.ndarray[DTYPE_t, ndim=2] U, V, P, Q, R
    cdef DTYPE_t A_L1 = np.linalg.norm(A, 1)

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
        for i in range(n):
            for j in range(n):
                A[i,j] = A[i,j] / 2 ** n_squarings
        U, V = _pade13(A)

    P = np.zeros((n, n), dtype=DTYPE)  # p_m(A) : numerator
    Q = np.zeros((n, n), dtype=DTYPE)  # q_m(A) : denominator
    for i in range(n):
        for j in range(n):
            P[i, j] = V[i, j] + U[i, j]
            Q[i, j] = V[i, j] - U[i, j]

    R = np.linalg.solve(Q, P)

    # squaring step to undo scaling
    for _ in range(n_squarings):
        R = np.dot(R, R)

    return R


cdef _pade3(np.ndarray[DTYPE_t, ndim=2] A):
    """ Helper method for expm

    Args:
        A (np.array): input Matrix
    Returns:
        tuple: Mystery Components
    """
    cdef int n = A.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] A2, U, V
    cdef np.ndarray[DTYPE_t, ndim=2] ident = square_identity(n)
    A2 = np.dot(A, A)
    U = np.dot(A,
               square_add(square_mult(A2, pade3_b3),
                          square_mult(ident, pade3_b1)))
    V = square_add(square_mult(A2, pade3_b2),
                   square_mult(ident, pade3_b0))
    return U, V


cdef _pade5(np.ndarray[DTYPE_t, ndim=2] A):
    """ Helper method for expm

    Args:
        A (np.array): input Matrix
    Returns:
        tuple: Mystery Components
    """
    n = A.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] ident = square_identity(n)
    cdef np.ndarray[DTYPE_t, ndim=2] A2, A4, U, V
    A2 = np.dot(A, A)
    A4 = np.dot(A2, A2)
    U = np.dot(A,
               square_tri_add(square_mult(A4, pade5_b5),
                              square_mult(A2, pade5_b3),
                              square_mult(ident, pade5_b1)))
    V = square_tri_add(square_mult(A4, pade5_b4),
                       square_mult(A2, pade5_b2),
                       square_mult(ident, pade5_b0))
    return U, V


cdef _pade7(np.ndarray[DTYPE_t, ndim=2] A):
    """ Helper method for expm

    Args:
        A (np.array): input Matrix
    Returns:
        tuple: Mystery Components
    """
    cdef int n = A.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] ident = square_identity(n)
    cdef np.ndarray[DTYPE_t, ndim=2] A2, A4, A6, U, V
    A2 = np.dot(A, A)
    A4 = np.dot(A2, A2)
    A6 = np.dot(A4, A2)
    U = np.dot(A,
               square_quad_add(square_mult(A6, pade7_b7),
                               square_mult(A4, pade7_b5),
                               square_mult(A2, pade7_b3),
                               square_mult(ident, pade7_b1)))
    V = square_quad_add(square_mult(A6, pade7_b6),
                        square_mult(A4, pade7_b4),
                        square_mult(A2, pade7_b2),
                        square_mult(ident, pade7_b0))
    return U, V


cdef _pade9(np.ndarray[DTYPE_t, ndim=2] A):
    """ Helper method for expm

    Args:
        A (np.array): input Matrix
    Returns:
        tuple: Mystery Components
    """
    cdef int n = A.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] A2, A4, A6, A8, U, V
    cdef np.ndarray[DTYPE_t, ndim=2] ident = square_identity(n)
    A2 = np.dot(A, A)
    A4 = np.dot(A2, A2)
    A6 = np.dot(A4, A2)
    A8 = np.dot(A6, A2)
    U = np.dot(A,
               square_add(square_tri_add(square_mult(A8, pade9_b9),
                                         square_mult(A6, pade9_b7),
                                         square_mult(A4, pade9_b5)),
                          square_add(square_mult(A2, pade9_b3),
                                     square_mult(ident, pade9_b1))))
    V = square_add(square_tri_add(square_mult(A8, pade9_b8),
                                  square_mult(A6 , pade9_b6),
                                  square_mult(A4, pade9_b4)),
                   square_add(square_mult(A2, pade9_b2),
                              square_mult(ident, pade9_b0)))
    return U, V


cdef _pade13(np.ndarray[DTYPE_t, ndim=2] A):
    """ Helper method for expm

    Args:
        A (np.array): input Matrix
    Returns:
        tuple: Mystery Components
    """
    cdef int n = A.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] ident = square_identity(n)
    cdef np.ndarray[DTYPE_t, ndim=2] A2, A4, A6, U, V
    A2 = np.dot(A, A)
    A4 = np.dot(A2, A2)
    A6 = np.dot(A4, A2)
    U = np.dot(A,
               square_add(np.dot(A6,
                                 square_tri_add(square_mult(A6, pade13_b13),
                                                square_mult(A4, pade13_b11),
                                                square_mult(A2, pade13_b9))),
                                 square_quad_add(square_mult(A6, pade13_b7),
                                                 square_mult(A4, pade13_b5),
                                                 square_mult(A2, pade13_b3),
                                                 square_mult(ident, pade13_b1))))
    V = square_add(np.dot(A6,
                          square_tri_add(square_mult(A6, pade13_b12),
                                         square_mult(A4, pade13_b10),
                                         square_mult(A2, pade13_b8))),
                   square_quad_add(square_mult(A6, pade13_b6),
                                   square_mult(A4, pade13_b4),
                                   square_mult(A2, pade13_b2),
                                   square_mult(ident, pade13_b0)))
    return U, V


cdef square_identity(int n):
    """ This function produces a square identity matrix

    Args:
        n (int): size of square matrix
    Returns:
        np.array: identity matrix
    """
    cdef np.ndarray[DTYPE_t, ndim=2] a = np.zeros((n, n), dtype=DTYPE)

    for i in range(n):
        a[i, i] = 1

    return a


cdef square_mult(np.ndarray[DTYPE_t, ndim=2] a, DTYPE_t m):
    """ TODO """
    int n = a.shape[0]
    np.ndarray[DTYPE_t, ndim=2] b = np.zeros((n, n), dtype=DTYPE_t)

    for i in range(n):
        for j in range(n):
            b[i, j] = a[i, j] * m

    return b


cdef square_add(np.ndarray[DTYPE_t, ndim=2] a, np.ndarray[DTYPE_t, ndim=2] b):
    """ TODO """
    int n = a.shape[0]
    np.ndarray[DTYPE_t, ndim=2] x = np.zeros((n, n), dtype=DTYPE_t)

    for i in range(n):
        for j in range(n):
            x[i, j] = a[i, j] + b[i, j]

    return x


cdef square_tri_add(np.ndarray[DTYPE_t, ndim=2] a, np.ndarray[DTYPE_t, ndim=2] b, np.ndarray[DTYPE_t, ndim=2] c):
    """ TODO """
    int n = a.shape[0]
    np.ndarray[DTYPE_t, ndim=2] x = np.zeros((n, n), dtype=DTYPE_t)

    for i in range(n):
        for j in range(n):
            x[i, j] = a[i, j] + b[i, j] + c[i, j]

    return x


cdef square_quad_add(np.ndarray[DTYPE_t, ndim=2] a, np.ndarray[DTYPE_t, ndim=2] b, np.ndarray[DTYPE_t, ndim=2] c, np.ndarray[DTYPE_t, ndim=2] d):
    """ TODO """
    int n = a.shape[0]
    np.ndarray[DTYPE_t, ndim=2] x = np.zeros((n, n), dtype=DTYPE_t)

    for i in range(n):
        for j in range(n):
            x[i, j] = a[i, j] + b[i, j] + c[i, j] + d[i, j]

    return x


"""
cdef square_dot(np.ndarray[DTYPE_t, ndim=2] a, np.ndarray[DTYPE_t, ndim=2] b):
    int n = a.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] x = np.zeros((n, n), dtype=DTYPE)

    x[0, 0] = a[0] * b[:, 0]
    x[1, 0] =
"""


# So many constants
cdef DTYPE_t pade3_b0 = 120.
cdef DTYPE_t pade3_b1 = 60.
cdef DTYPE_t pade3_b2 = 12.
cdef DTYPE_t pade3_b3 = 1.

cdef DTYPE_t pade5_b0 = 30240.
cdef DTYPE_t pade5_b1 = 15120.
cdef DTYPE_t pade5_b2 = 3360.
cdef DTYPE_t pade5_b3 = 420.
cdef DTYPE_t pade5_b4 = 30.
cdef DTYPE_t pade5_b5 = 1.

cdef DTYPE_t pade7_b0 = 17297280.
cdef DTYPE_t pade7_b1 = 8648640.
cdef DTYPE_t pade7_b3 = 1995840.
cdef DTYPE_t pade7_b3 = 277200.
cdef DTYPE_t pade7_b4 = 25200.
cdef DTYPE_t pade7_b5 = 1512.
cdef DTYPE_t pade7_b6 = 56.
cdef DTYPE_t pade7_b7 = 1.

cdef DTYPE_t pade9_b0 = 17643225600.
cdef DTYPE_t pade9_b1 = 8821612800.
cdef DTYPE_t pade9_b2 = 2075673600.
cdef DTYPE_t pade9_b3 = 302702400.
cdef DTYPE_t pade9_b4 = 30270240.
cdef DTYPE_t pade9_b5 = 2162160.
cdef DTYPE_t pade9_b6 = 110880.
cdef DTYPE_t pade9_b7 = 3960.
cdef DTYPE_t pade9_b8 = 90.
cdef DTYPE_t pade9_b9 = 1.

cdef DTYPE_t pade13_b0 = 64764752532480000.
cdef DTYPE_t pade13_b1 = 32382376266240000.
cdef DTYPE_t pade13_b2 = 7771770303897600.
cdef DTYPE_t pade13_b3 = 1187353796428800.
cdef DTYPE_t pade13_b4 = 129060195264000.
cdef DTYPE_t pade13_b5 = 10559470521600.
cdef DTYPE_t pade13_b6 = 670442572800.
cdef DTYPE_t pade13_b7 = 33522128640.
cdef DTYPE_t pade13_b8 = 1323241920.
cdef DTYPE_t pade13_b9 = 40840800.
cdef DTYPE_t pade13_b10 = 960960.
cdef DTYPE_t pade13_b11 = 16380.
cdef DTYPE_t pade13_b12 = 182.
cdef DTYPE_t pade13_b13 = 1.
