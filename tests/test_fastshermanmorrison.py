import numpy as np
import unittest

from fastshermanmorrison.fastshermanmorrison import ShermanMorrison
from fastshermanmorrison.fastshermanmorrison import FastShermanMorrison


class ShermanMorrisonRef(object):
    """Reference container class for Sherman-morrison array inversion.

    This version uses the tried-and-true slice index formulation
    """

    def __init__(self, jvec, slices, nvec=0.0):
        self._jvec = jvec
        self._slices = slices
        self._nvec = nvec

    def __add__(self, other):
        nvec = self._nvec + other
        return ShermanMorrisonRef(self._jvec, self._slices, nvec)

    # hacky way to fix adding 0
    def __radd__(self, other):
        if other == 0:
            return self.__add__(other)
        else:
            raise TypeError

    def _solve_D1(self, x):
        """Solves :math:`N^{-1}x` where :math:`x` is a vector."""

        Nx = x / self._nvec
        for slc, jv in zip(self._slices, self._jvec):
            if slc.stop - slc.start > 1:
                rblock = x[slc]
                niblock = 1 / self._nvec[slc]
                beta = 1.0 / (np.einsum("i->", niblock) + 1.0 / jv)
                Nx[slc] -= beta * np.dot(niblock, rblock) * niblock
        return Nx

    def _solve_1D1(self, x, y):
        """Solves :math:`y^T N^{-1}x`, where :math:`x` and
        :math:`y` are vectors.
        """

        Nx = x / self._nvec
        yNx = np.dot(y, Nx)
        for slc, jv in zip(self._slices, self._jvec):
            if slc.stop - slc.start > 1:
                xblock = x[slc]
                yblock = y[slc]
                niblock = 1 / self._nvec[slc]
                beta = 1.0 / (np.einsum("i->", niblock) + 1.0 / jv)
                yNx -= beta * np.dot(niblock, xblock) * np.dot(niblock, yblock)
        return yNx

    def _solve_2D2(self, X, Z):
        """Solves :math:`Z^T N^{-1}X`, where :math:`X`
        and :math:`Z` are 2-d arrays.
        """

        ZNX = np.dot(Z.T / self._nvec, X)
        for slc, jv in zip(self._slices, self._jvec):
            if slc.stop - slc.start > 1:
                Zblock = Z[slc, :]
                Xblock = X[slc, :]
                niblock = 1 / self._nvec[slc]
                beta = 1.0 / (np.einsum("i->", niblock) + 1.0 / jv)
                zn = np.dot(niblock, Zblock)
                xn = np.dot(niblock, Xblock)
                ZNX -= beta * np.outer(zn.T, xn)
        return ZNX

    def _get_logdet(self):
        """Returns log determinant of :math:`N+UJU^{T}` where :math:`U`
        is a quantization matrix.
        """
        logdet = np.einsum("i->", np.log(self._nvec))
        for slc, jv in zip(self._slices, self._jvec):
            if slc.stop - slc.start > 1:
                niblock = 1 / self._nvec[slc]
                beta = 1.0 / (np.einsum("i->", niblock) + 1.0 / jv)
                logdet += np.log(jv) - np.log(beta)
        return logdet

    def solve(self, other, left_array=None, logdet=False):
        if other.ndim == 1:
            if left_array is None:
                ret = self._solve_D1(other)
            elif left_array is not None and left_array.ndim == 1:
                ret = self._solve_1D1(other, left_array)
            elif left_array is not None and left_array.ndim == 2:
                ret = np.dot(left_array.T, self._solve_D1(other))
            else:
                raise TypeError
        elif other.ndim == 2:
            if left_array is None:
                raise NotImplementedError(
                    "ShermanMorrisonRef does not implement _solve_D2"
                )
            elif left_array is not None and left_array.ndim == 2:
                ret = self._solve_2D2(other, left_array)
            elif left_array is not None and left_array.ndim == 1:
                ret = np.dot(other.T, self._solve_D1(left_array))
            else:
                raise TypeError
        else:
            raise TypeError

        return (ret, self._get_logdet()) if logdet else ret


class TestFastShermanMorrison(unittest.TestCase):
    def get_test_data(self):
        """Set up test data that we use in the tests"""

        # Observing times and 'random' vector
        x = np.array([1.0, -1.0, 0.0, 3.0, -2.0, 1.0, -1.0, 4.0, 1.0, -1.0])
        x2 = x[::-1]

        # Some 2D test matrices
        X = np.vstack([x, x2]).T
        Z = X.T.flatten().reshape(*X.shape)

        return x, x2, X, Z

    def get_test_quantities(self):
        """Set up test quantities that we use in all tests"""

        # Slices defining epochs (note the 5th index is left out)
        slices = [slice(0, 5), slice(6, 10)]

        # Nvec is constant here
        nvec = np.ones(10)

        # Jvec is the jitter amplitude
        jvec = np.array([1.0, 2.0])

        return slices, nvec, jvec

    def get_index_arrays(self, slices):
        """Introduce an arbitrary ordering in the data"""

        slc_inds = [np.arange(slc.start, slc.stop) for slc in slices]

        isort = np.array([6, 2, 0, 9, 3, 4, 1, 8, 7, 5])
        iisort = np.zeros_like(isort)
        for ii, pp in enumerate(isort):
            iisort[pp] = ii

        # isort  = [6, 2, 0, 9, 3, 4, 1, 8, 7, 5]
        # iisort = [2, 6, 1, 4, 5, 9, 0, 8, 7, 3]
        #
        # index_arrays = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
        # should become: [[2, 6, 1, 4, 5], [9, 0, 8, 7, 3]]
        # that is: [[iisort[idx] for idx in idxs] for idxs in index_arrays]

        idxs = [np.array([iisort[idx] for idx in idxs]) for idxs in slc_inds]

        # return isort, index_arrays[iisort]
        return isort, iisort, idxs

    def get_sm_objects(self):
        """Get all the ShermanMorrison objects"""

        slices, nvec, jvec = self.get_test_quantities()

        # Create all the regular ShermanMorrison objects
        smr = ShermanMorrisonRef(jvec, slices, nvec=nvec)
        sm = ShermanMorrison(jvec, slices, nvec=nvec)
        fsm = FastShermanMorrison(jvec, slices, nvec=nvec)

        return smr, sm, fsm

    def get_shuffled_sm_objects(self):
        """Get all the 'sorted' ShermanMorrison objects"""

        slices, nvec, jvec = self.get_test_quantities()
        isort, iisort, idxs = self.get_index_arrays(slices)

        # Create sorted ShermanMorrison objects
        sms = ShermanMorrison(jvec, idxs, nvec=nvec[isort])
        fsms = FastShermanMorrison(jvec, idxs, nvec=nvec[isort])

        return sms, fsms, isort, iisort

    def test_solve_D1(self):
        """Test the D1 solve routines"""

        x, _, _, _ = self.get_test_data()
        smr, sm, fsm = self.get_sm_objects()
        sms, fsms, isort, iisort = self.get_shuffled_sm_objects()

        # Regular ShermanMorrison, with slice objects
        self.assertTrue(np.allclose(smr.solve(x), sm.solve(x)))

        # Fast ShermanMorrison, with slice objects
        self.assertTrue(np.allclose(smr.solve(x), fsm.solve(x)))

        # Regular SermanMorrison, shuffled data
        self.assertTrue(np.allclose(smr.solve(x), sms.solve(x[isort])[iisort]))

        # Fast SermanMorrison, shuffled data
        self.assertTrue(np.allclose(smr.solve(x), fsms.solve(x[isort])[iisort]))

    def test_solve_1D1(self):
        """Test the 1D1 solve routines"""

        x, x2, _, _ = self.get_test_data()
        smr, sm, fsm = self.get_sm_objects()
        sms, fsms, isort, _ = self.get_shuffled_sm_objects()

        # Regular ShermanMorrison, with slice objects
        self.assertEqual(smr.solve(x, x2), sm.solve(x, x2))

        # Fast ShermanMorrison, with slice objects
        self.assertEqual(smr.solve(x, x2), fsm.solve(x, x2))

        # Regular ShermanMorrison, shuffled data
        self.assertEqual(smr.solve(x, x2), sms.solve(x[isort], x2[isort]))

        # Fast ShermanMorrison, shuffled data
        self.assertEqual(smr.solve(x, x2), fsms.solve(x[isort], x2[isort]))

    def test_solve_2D2(self):
        """Test the 2D2 solve routines"""

        _, _, X, Z = self.get_test_data()
        smr, sm, fsm = self.get_sm_objects()
        sms, fsms, isort, _ = self.get_shuffled_sm_objects()

        # Regular ShermanMorrison, with slice objects
        self.assertTrue(np.allclose(smr.solve(X, Z), sm.solve(X, Z)))

        # Fast ShermanMorrison, with slice objects
        self.assertTrue(np.allclose(smr.solve(X, Z), fsm.solve(X, Z)))

        # Regular ShermanMorrison, shuffled data
        self.assertTrue(
            np.allclose(smr.solve(X, Z), sms.solve(X[isort, :], Z[isort, :]))
        )

        # Fast ShermanMorrison, shuffled data
        self.assertTrue(
            np.allclose(smr.solve(X, Z), fsms.solve(X[isort, :], Z[isort, :]))
        )

    def test_solve_D2(self):
        """Test the D2 solve routines (exceptions)"""

        _, _, X, _ = self.get_test_data()
        _, sm, fsm = self.get_sm_objects()
        sms, fsms, isort, _ = self.get_shuffled_sm_objects()

        with self.assertRaises(NotImplementedError):
            sm.solve(X)

        with self.assertRaises(NotImplementedError):
            fsm.solve(X)

        with self.assertRaises(NotImplementedError):
            sms.solve(X)

        with self.assertRaises(NotImplementedError):
            fsms.solve(X)

    def test_errors(self):
        """Test the exceptions in te classes"""

        x, _, X, _ = self.get_test_data()
        _, sm, fsm = self.get_sm_objects()
        sms, fsms, _, _ = self.get_shuffled_sm_objects()

        mat3d = np.zeros((2, 2, 2))

        with self.assertRaises(TypeError):
            sm.solve(x, mat3d)

        with self.assertRaises(TypeError):
            sm.solve(X, mat3d)

        with self.assertRaises(TypeError):
            sm.solve(mat3d)

        with self.assertRaises(TypeError):
            fsm.solve(x, mat3d)

        with self.assertRaises(TypeError):
            fsm.solve(X, mat3d)

        with self.assertRaises(TypeError):
            fsm.solve(mat3d)

        with self.assertRaises(TypeError):
            sms.solve(x, mat3d)

        with self.assertRaises(TypeError):
            sms.solve(X, mat3d)

        with self.assertRaises(TypeError):
            sms.solve(mat3d)

        with self.assertRaises(TypeError):
            fsms.solve(x, mat3d)

        with self.assertRaises(TypeError):
            fsms.solve(X, mat3d)

        with self.assertRaises(TypeError):
            fsms.solve(mat3d)

    def test_logdet(self):
        x, _, _, _ = self.get_test_data()
        smr, sm, fsm = self.get_sm_objects()
        sms, fsms, _, _ = self.get_shuffled_sm_objects()

        _, smr_ld = smr.solve(x, logdet=True)
        _, sm_ld = sm.solve(x, logdet=True)
        _, fsm_ld = fsm.solve(x, logdet=True)
        _, sms_ld = sms.solve(x, logdet=True)
        _, fsms_ld = fsms.solve(x, logdet=True)

        self.assertEqual(smr_ld, sm_ld)

        self.assertEqual(smr_ld, fsm_ld)

        self.assertEqual(smr_ld, sms_ld)

        self.assertEqual(smr_ld, fsms_ld)


if __name__ == "__main__":
    unittest.main()
