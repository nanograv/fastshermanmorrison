import numpy as np
from . import cython_fastshermanmorrison as cfsm


def indices_from_slice(slc):
    """Given a list of slice objects, return a list of index arrays"""

    if isinstance(slc, np.ndarray):
        return slc
    else:
        return np.arange(*slc.indices(slc.stop))


def pack_indices(idxs):
    """Given a list of indices, provide the Cython tight packing

    :param idxs: List of NumPy arrays containing indices for each ECORR block
    :return: sorting indices, and Uinds of the blocks
    """

    # Concatenate all the index arrays and get their original positions
    slc_isort = np.concatenate(idxs)

    # Generate new Uinds with start and stop indices for each block in indices
    lengths = [len(x) for x in idxs]
    starts = np.cumsum([0] + lengths[:-1])
    stops = np.cumsum(lengths)

    new_Uinds = np.column_stack([starts, stops])

    return slc_isort, new_Uinds


def extend_isort(isort, length):
    """Extend the isort index array if necessary"""

    leftover = list(set(np.arange(length)) - set(isort))
    return np.concatenate([isort, leftover])


class ShermanMorrison(object):
    """Custom container class for Sherman-morrison array inversion."""

    def __init__(self, jvec, slices, nvec=0.0):
        self._jvec = jvec
        self._slices = slices
        self._idxs = [indices_from_slice(slc) for slc in slices]
        self._nvec = nvec

    def __add__(self, other):
        nvec = self._nvec + other
        return ShermanMorrison(self._jvec, self._slices, nvec)

    # hacky way to fix adding 0
    def __radd__(self, other):
        if other == 0:
            return self.__add__(other)
        else:
            raise TypeError

    def _solve_D1(self, x):
        """Solves :math:`N^{-1}x` where :math:`x` is a vector."""

        Nx = x / self._nvec
        for idx, jv in zip(self._idxs, self._jvec):
            if len(idx) > 1:
                rblock = x[idx]
                niblock = 1 / self._nvec[idx]
                beta = 1.0 / (np.einsum("i->", niblock) + 1.0 / jv)
                Nx[idx] -= beta * np.dot(niblock, rblock) * niblock
        return Nx

    def _solve_1D1(self, x, y):
        """Solves :math:`y^T N^{-1}x`, where :math:`x` and
        :math:`y` are vectors.
        """

        Nx = x / self._nvec
        yNx = np.dot(y, Nx)
        for idx, jv in zip(self._idxs, self._jvec):
            if len(idx) > 1:
                xblock = x[idx]
                yblock = y[idx]
                niblock = 1 / self._nvec[idx]
                beta = 1.0 / (np.einsum("i->", niblock) + 1.0 / jv)
                yNx -= beta * np.dot(niblock, xblock) * np.dot(niblock, yblock)
        return yNx

    def _solve_2D2(self, X, Z):
        """Solves :math:`Z^T N^{-1}X`, where :math:`X`
        and :math:`Z` are 2-d arrays.
        """

        ZNX = np.dot(Z.T / self._nvec, X)
        for idx, jv in zip(self._idxs, self._jvec):
            if len(idx) > 1:
                Zblock = Z[idx, :]
                Xblock = X[idx, :]
                niblock = 1 / self._nvec[idx]
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
        for idx, jv in zip(self._idxs, self._jvec):
            if len(idx) > 1:
                niblock = 1 / self._nvec[idx]
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
                    "ShermanMorrison does not implement _solve_D2"
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


class FastShermanMorrison(ShermanMorrison):
    """Custom container class for Sherman-morrison array inversion."""

    def __init__(self, jvec, slices, nvec=0.0):
        """Initialize the Fast Sherman-Morrison object"""

        try:
            self._uinds = np.vstack([[slc.start, slc.stop] for slc in slices])
            self._as_slice = True
        except AttributeError:
            self._idxs = [indices_from_slice(slc) for slc in slices]
            self._slc_isort, self._uinds = pack_indices(self._idxs)
            self._as_slice = False

        super().__init__(jvec, slices, nvec=nvec)

    def __add__(self, other):
        nvec = self._nvec + other
        return FastShermanMorrison(self._jvec, self._idxs, nvec)

    def _solve_D1(self, x):
        """Solves :math:`N^{-1}x` where :math:`x` is a vector."""

        if self._as_slice:
            return cfsm.cython_block_shermor_0D(x, self._nvec, self._jvec, self._uinds)
        else:
            return cfsm.cython_idx_block_shermor_0D(
                x, self._nvec, self._jvec, self._uinds, self._slc_isort
            )

    def _solve_1D1(self, x, y):
        """Solves :math:`y^T N^{-1}x`, where :math:`x` and
        :math:`y` are vectors.
        """

        if self._as_slice:
            _, yNx = cfsm.cython_block_shermor_1D1(
                x, y, self._nvec, self._jvec, self._uinds
            )
        else:
            _, yNx = cfsm.cython_idx_block_shermor_1D1(
                x, y, self._nvec, self._jvec, self._uinds, self._slc_isort
            )

        return yNx

    def _solve_2D2(self, X, Z):
        """Solves :math:`Z^T N^{-1}X`, where :math:`X`
        and :math:`Z` are 2-d arrays.
        """

        if self._as_slice:
            _, ZNX = cfsm.cython_blas_block_shermor_2D_asymm(
                Z, X, self._nvec, self._jvec, self._uinds
            )
        else:
            if len(self._slc_isort) < Z.shape[0]:
                self._slc_isort = extend_isort(self._slc_isort, Z.shape[0])

            _, ZNX = cfsm.cython_blas_idx_block_shermor_2D_asymm(
                Z, X, self._nvec, self._jvec, self._uinds, self._slc_isort
            )
        return ZNX

    def _get_logdet(self):
        """Returns log determinant of :math:`N+UJU^{T}` where :math:`U`
        is a quantization matrix.
        """

        if self._as_slice:
            logJdet, _ = cfsm.cython_block_shermor_1D(
                np.zeros_like(self._nvec), self._nvec, self._jvec, self._uinds
            )
        else:
            logJdet, _ = cfsm.cython_idx_block_shermor_1D(
                np.zeros_like(self._nvec),
                self._nvec,
                self._jvec,
                self._uinds,
                self._slc_isort,
            )

        return logJdet

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
                    "FastShermanMorrison does not implement _solve_D2"
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
