cimport cython
from numpy cimport ndarray


@cython.locals(A_L1=cython.double, n_squarings=cython.int, U=ndarray, V=ndarray, P=ndarray, Q=ndarray, R=ndarray)
cpdef ndarray expm(ndarray A)


@cython.locals(b=tuple, shape=tuple, ident=ndarray, A2=ndarray, U=ndarray, V=ndarray)
cdef tuple _pade3(ndarray A)


@cython.locals(b=tuple, shape=tuple, ident=ndarray, A2=ndarray, A4=ndarray, U=ndarray, V=ndarray)
cdef tuple _pade5(ndarray A)


@cython.locals(b=tuple, shape=tuple, ident=ndarray, A2=ndarray, A4=ndarray, A6=ndarray, U=ndarray, V=ndarray)
cdef tuple _pade7(ndarray A)


@cython.locals(b=tuple, shape=tuple, ident=ndarray, A2=ndarray, A4=ndarray, A6=ndarray, A8=ndarray, U=ndarray, V=ndarray)
cdef tuple _pade9(ndarray A)


@cython.locals(b=tuple, shape=tuple, ident=ndarray, A2=ndarray, A4=ndarray, A6=ndarray, U=ndarray, V=ndarray)
cdef tuple _pade13(ndarray A)
