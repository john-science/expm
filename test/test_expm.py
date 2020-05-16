import numpy as np
import unittest
from expm import expm
from scipy.linalg import expm as expm_scipy
import scipy.sparse as sparse


class ExpMTest(unittest.TestCase):

    def test_expm_random_sparse(self):
        """ Testing against a large set of random 2D matricies with defined sparsity
        """
        np.random.seed(7)

        for size in range(5, 40):
            for d in [x / 100 for x in range(10, 90)]:
                mat = sparse.random(size, size, density=d).toarray()
                phi = expm(mat)
                out_scipy = expm_scipy(mat)
                assert_arrays_equal(out_scipy, phi, decimal=7)

    def test_poorly_conditioned(self):
        """ Testing against a single poorly-conditioned matrix.
        Generated as a random 6x6 array, and then divided the first row and column by 1e6
        """
        mat = np.array([[6.6683264e-07, 6.3376014e-07, 4.5754601e-07, 7.1252750e-07, 7.5942456e-07, 5.9028134e-07],
                        [5.7484930e-07, 6.5606222e-01, 6.5073522e-01, 2.4251825e-01, 1.0735555e-01, 2.6956707e-01],
                        [6.1098206e-07, 4.0445840e-01, 5.6152644e-01, 5.8278476e-01, 8.0418942e-01, 7.7306821e-01],
                        [8.5656473e-07, 4.3833014e-01, 5.7838875e-01, 2.4317287e-01, 1.6641302e-01, 3.9590950e-01],
                        [4.4858020e-07, 1.1731434e-01, 7.3598305e-01, 4.5670520e-01, 5.8185932e-01, 9.4438709e-01],
                        [4.1073805e-07, 9.6286350e-02, 7.7365790e-01, 1.3120009e-01, 9.3908360e-01, 1.4665616e-01]])

        phi = expm(mat)
        out_scipy = expm_scipy(mat)
        assert_arrays_equal(out_scipy, phi, decimal=8)

    def test_expm_all_zeros(self):
        """ Test against a matrix of all zeros """
        a = np.zeros((20, 20), dtype='float32')
        phi = expm(a)
        out_scipy = expm_scipy(a)
        assert_arrays_equal(out_scipy, phi, decimal=12)

    def test_expm_all_ones(self):
        """ Test against a matrix of all ones """
        a = np.ones((33, 33), dtype='float64')
        phi = expm(a)
        out_scipy = expm_scipy(a)
        assert_arrays_equal(out_scipy, phi, decimal=12)

    def test_expm_random_integers(self):
        """ a simple, arbitrary matrix """
        a = np.array([[1, 2, 3], [1, 2, 1.1], [1, 0, 3]], dtype=np.float64)
        phi = expm(a)
        out_scipy = expm_scipy(a)
        assert_arrays_equal(out_scipy, phi, decimal=12)

    def test_expm_one_large_element(self):
        """ one value slightly swamping the rest """
        a = np.array([[0.1, 0.02, 0.003], [1.875, 0.12, 0.11], [0.1234567, 0, 0.3]], dtype=np.float64)
        phi = expm(a)
        out_scipy = expm_scipy(a)
        assert_arrays_equal(out_scipy, phi, decimal=12)

    def test_expm_one_huge_element(self):
        """ one value swamping the rest """
        a = np.array([[999.875, 0.2, 0.3], [0.1, 0.002, 0.001], [0.01, 0, -0.003]], dtype=np.float64)
        phi = expm(a)
        out_scipy = expm_scipy(a)
        assert_arrays_equal(out_scipy, phi, decimal=12)

    def test_expm_nearly_identity(self):
        """ nearly the identity matrix """
        a = np.array([[1.0012, 0, 0.0003], [0.0075, 0.9876543, 0.0011], [0, 0, 0.9873]], dtype=np.float64)
        phi = expm(a)
        out_scipy = expm_scipy(a)
        assert_arrays_equal(out_scipy, phi, decimal=12)

    def test_expm_random_non_zero_floats(self):
        """ arbitrarily large values, and some negatives """
        a = np.array([[99.23452, 2.0000234523, 0.0003, 378.2362], [1.00001, 8754.236, 1.1007, 33.333333],
                      [111, 0.00034, 3922.323, -999.333], [-1234.5678, -0.00034, 333.65, 13]], dtype=np.float64)
        phi = expm(a)
        out_scipy = expm_scipy(a)
        assert_arrays_equal(out_scipy, phi, decimal=12)

    def test_expm_smallish_floats(self):
        """ This is just some arbitrary set of small-ish floats """
        a = np.arange(1, 17, dtype='float64').reshape(4, 4) / 1e3
        phi = expm(a)
        out_scipy = expm_scipy(a)
        assert_arrays_equal(out_scipy, phi, decimal=12)


def array_to_csv(a):
    """ returns a custom CSV string version of NumPy (1D or 2D) array

    Args:
    	a (np.array): input 1D or 2D numpy array
    Returns:
    	str: string representation of the array as a CSV
    """
    def num2str(v):
        """ returns a custom string version of a number """
        s = '%.16f' % v
        if '.' in s:
            s = s.split('.')[0] + '.' + s.split('.')[1].rstrip('0')
        if s[-1] == '.':
            s = s[:-1]
        return s

    # if necessary, cast a list to an array
    if type(a) not in (np.ndarray, np.array):
        a = np.array(a)

    # build the CSV string
    csv = []
    if len(a.shape) == 1:
        for val in a:
            csv.append(num2str(val))
    elif len(a.shape) == 2:
        for row in a:
            csv.append(','.join([num2str(val) for val in row]))
    else:
        raise ValueError('ERROR: Array shape out of bounds: ' + str(a.shape))

    return '\n'.join(csv)


def assert_arrays_equal(i1, i2, decimal=12, err_msg="arrays not equal"):
    """ Unit Testing Helper Function
    Tests if two numpy arrays are equal using absolute precision.
    If not, prints them as CSVs.
    """
    a1 = i1.flatten()
    a2 = i2.flatten()

    try:
        assert len(a1.flatten()) == len(a2)
    except:
        print('Array lengths not equal: {0} != {1}'.format(len(a1), len(a2)))
        print(err_msg)
        raise

    try:
        np.testing.assert_array_almost_equal(a1, a2, decimal=decimal)
    except AssertionError as e:
        print(e)
        print('\nArray 1:\n')
        print(array_to_csv(a1))
        print('\nArray 2:\n')
        print(array_to_csv(a2))
        for i in range(len(a1)):
            try:
                np.testing.assert_array_almost_equal(a1[i], a2[i], decimal=decimal)
            except:
                print('\nArray index {0} is first non-equal element: {1} != {2}'.format(i, float(a1[i]), a2[i]))
                break
        print(err_msg)
        raise


if __name__ == '__main__':
    unittest.main()
