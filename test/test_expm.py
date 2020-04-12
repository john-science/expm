import numpy as np
import unittest
from expm import expm
from scipy.linalg import expm as expm_scipy


class ExpMTest(unittest.TestCase):

    def test_expm_0(self):
        """ Test against a matrix of all zeros """
        a = np.zeros((20, 20), dtype='float32')
        phi = expm(a)
        out_scipy = expm_scipy(a)
        assert_arrays_equal(out_scipy, phi, decimal=12)

    def test_expm_1(self):
        """ Test against a matrix of all ones """
        a = np.ones((33, 33), dtype='float64')
        phi = expm(a)
        out_scipy = expm_scipy(a)
        assert_arrays_equal(out_scipy, phi, decimal=12)

    def test_expm_2(self):
        """ a simple, arbitrary matrix """
        a = np.array([[1, 2, 3], [1, 2, 1.1], [1, 0, 3]], dtype=np.float64)
        phi = expm(a)
        out_scipy = expm_scipy(a)
        assert_arrays_equal(out_scipy, phi, decimal=12)

    def test_expm_3(self):
        """ one value slightly swamping the rest """
        a = np.array([[0.1, 0.02, 0.003], [1.875, 0.12, 0.11], [0.1234567, 0, 0.3]], dtype=np.float64)
        phi = expm(a)
        out_scipy = expm_scipy(a)
        assert_arrays_equal(out_scipy, phi, decimal=12)

    def test_expm_4(self):
        """ one value swamping the rest """
        all = np.array([[999.875, 0.2, 0.3], [0.1, 0.002, 0.001], [0.01, 0, -0.003]], dtype=np.float64)
        phi = expm(a)
        out_scipy = expm_scipy(a)
        assert_arrays_equal(out_scipy, phi, decimal=12)

    def test_expm_4(self):
        """ nearly the identify matrix """
        a = np.array([[1.0012, 0, 0.0003], [0.0075, 0.9876543, 0.0011], [0, 0, 0.9873]], dtype=np.float64)
        phi = expm(a)
        out_scipy = expm_scipy(a)
        assert_arrays_equal(out_scipy, phi, decimal=12)

    def test_expm_5(self):
        """ arbitrarily large values, and some negatives """
        a = np.array([[99.23452, 2.0000234523, 0.0003, 378.2362], [1.00001, 8754.236, 1.1007, 33.333333],
                      [111, 0.00034, 3922.323, -999.333], [-1234.5678, -0.00034, 333.65, 13]], dtype=np.float64)
        phi = expm(a)
        out_scipy = expm_scipy(a)
        assert_arrays_equal(out_scipy, phi, decimal=12)

    def test_expm_6(self):
        """ This is just some arbitrary set of integers """
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
