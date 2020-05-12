cimport cython
from numpy cimport ndarray, float_t


ctypedef float_t DTYPE_t


@cython.locals(A_L1=DTYPE_t, n_squarings=int, U=ndarray, V=ndarray, P=ndarray, Q=ndarray, R=ndarray)
cpdef ndarray expm(ndarray A)


@cython.locals(n=int, ident=ndarray, A2=ndarray, U=ndarray, V=ndarray)
cdef tuple _pade3(ndarray A)


@cython.locals(n=int, ident=ndarray, A2=ndarray, A4=ndarray, U=ndarray, V=ndarray)
cdef tuple _pade5(ndarray A)


@cython.locals(n=int, ident=ndarray, A2=ndarray, A4=ndarray, A6=ndarray, U=ndarray, V=ndarray)
cdef tuple _pade7(ndarray A)


@cython.locals(n=int, ident=ndarray, A2=ndarray, A4=ndarray, A6=ndarray, A8=ndarray, U=ndarray, V=ndarray)
cdef tuple _pade9(ndarray A)


@cython.locals(n=int, ident=ndarray, A2=ndarray, A4=ndarray, A6=ndarray, U=ndarray, V=ndarray)
cdef tuple _pade13(ndarray A)


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
cdef DTYPE_t pade7_b2 = 1995840.
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

