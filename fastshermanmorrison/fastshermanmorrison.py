import numpy as np
import scipy.linalg as sl
from collections import defaultdict
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
        self._has_sqrtsolve = True

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

    def _sqrtsolve_D2(self, x):
        """Solves :math:`N^{-1/2}x` where :math:`x` is a 2-d array."""

        Lix = x / np.sqrt(self._nvec)[:, None]
        for idx, jv in zip(self._idxs, self._jvec):
            Xblock = x[idx, :]
            Nblock = np.diag(self._nvec[idx])
            Nblock += jv * np.ones_like(Nblock)
            Lblock = sl.cholesky(Nblock, lower=True)
            Lix[idx, :] = sl.solve_triangular(Lblock, Xblock, trans=0, lower=True)

        return Lix

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

    def sqrtsolve(self, other, left_array=None):
        if other.ndim == 1:
            shape = other.shape
            ret = self._sqrtsolve_D2(other.reshape(-1, 1)).reshape(*shape)

            if left_array is not None and left_array.ndim == 1:
                ret = np.sum(left_array * ret)
            elif left_array is not None:
                raise NotImplementedError(
                    "ShermanMorrison does not implement _sqrtsolve_1D2"
                )
        elif other.ndim == 2:
            if left_array is None:
                ret = self._sqrtsolve_D2(other)
            elif left_array is not None and left_array.ndim == 2:
                raise NotImplementedError(
                    "ShermanMorrison does not implement _sqrtsolve_2D2"
                )
            elif left_array is not None and left_array.ndim == 1:
                raise NotImplementedError(
                    "ShermanMorrison does not implement _sqrtsolve_1D2"
                )
            else:
                raise TypeError
        else:
            raise TypeError

        return ret


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

    def _sqrtsolve_D2(self, x):
        """
        Block‑wise solve   L_block^{-1} X_block
        for each N_block = diag(d) + j * 1 1^T,
        where L_block L_block^T = N_block,
        using a true Cholesky rank‑1 update + forward triangular solve.

        Parameters
        ----------
        X : ndarray, shape (n, ℓ)
            Right‑hand sides.

        Returns
        -------
        Nx : ndarray, shape (n, ℓ)
            L^{-1} X.
        """
        if self._as_slice:
            Lix = cfsm.cython_block_sqrtsolve_rank1(
                x, self._nvec, self._jvec, self._uinds
            )
        else:
            Lix = cfsm.cython_idx_sqrtsolve_rank1(
                x, self._nvec, self._jvec, self._uinds, self._slc_isort
            )

        return Lix

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

    def sqrtsolve(self, other, left_array=None):
        if other.ndim == 1:
            shape = other.shape
            ret = self._sqrtsolve_D2(other.reshape(-1, 1)).reshape(*shape)

            if left_array is not None and left_array.ndim == 1:
                ret = np.sum(left_array * ret)
            elif left_array is not None:
                raise NotImplementedError(
                    "ShermanMorrison does not implement _sqrtsolve_1D2"
                )
        elif other.ndim == 2:
            if left_array is None:
                ret = self._sqrtsolve_D2(other)
            elif left_array is not None and left_array.ndim == 2:
                raise NotImplementedError(
                    "ShermanMorrison does not implement _sqrtsolve_2D2"
                )
            elif left_array is not None and left_array.ndim == 1:
                raise NotImplementedError(
                    "ShermanMorrison does not implement _sqrtsolve_1D2"
                )
            else:
                raise TypeError
        else:
            raise TypeError

        return ret


class ShermanMorrisonRK(object):
    """Diagonal + overlapping-blocks rank-k Woodbury updates."""

    def __init__(self, jvec, slices, nvec):
        self._nvec = np.asarray(nvec)
        self._orig_idxs = [indices_from_slice(s) for s in slices]
        self._jvec      = np.asarray(jvec)
        self._has_sqrtsolve = True

        # Disjoint-set (union-find) initialization
        B = len(self._orig_idxs)
        parent = list(range(B))
        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i, j):
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[rj] = ri

        # Inverted index: map each position → list of blocks that contain it
        pos2blocks = defaultdict(list)
        for b, idx in enumerate(self._orig_idxs):
            for p in idx:
                pos2blocks[p].append(b)

        # Union all blocks that share any position
        for blocks in pos2blocks.values():
            first = blocks[0]
            for b in blocks[1:]:
                union(first, b)

        # Gather connected components of block-IDs
        comps = {}
        for b in range(B):
            root = find(b)
            comps.setdefault(root, []).append(b)

        # assemble per-component data
        self._components = []
        for comp_blocks in comps.values():
            # union all indices
            all_idx = sorted({i for b in comp_blocks for i in self._orig_idxs[b]})
            pos_map = {idx:i for i, idx in enumerate(all_idx)}

            k = len(comp_blocks)
            D = len(all_idx)
            U = np.zeros((D, k))
            comp_j = np.zeros(k, dtype=np.double)
            for col, b in enumerate(sorted(comp_blocks)):
                comp_j[col] = self._jvec[b]
                for i in self._orig_idxs[b]:
                    U[pos_map[i], col] = 1.0

            self._components.append({
                'idxs': np.array(all_idx, dtype=int),
                'U'   : U,
                'j'   : comp_j,
            })

    def _solve_D1(self, x):
        # first: diagonal solve
        out = x / self._nvec
        # then subtract each component’s low-rank correction
        for comp in self._components:
            idxs = comp['idxs']
            U    = comp['U']
            j    = comp['j']
            Dinv = 1.0 / self._nvec[idxs]

            # build M = J^{-1} + U^T (Dinv * U)
            Jinv = np.diag(1.0 / j)
            M    = Jinv + U.T.dot(Dinv[:,None] * U)
            invM = np.linalg.inv(M)

            # k = U^T (Dinv * x[idxs])
            k_vec = U.T.dot(Dinv * x[idxs])
            # correction = Dinv * ( U ( invM @ k_vec ) )
            corr = Dinv * ( U.dot(invM.dot(k_vec)) )
            out[idxs] -= corr
        return out

    def _solve_1D1(self, x, y):
        # y^T (N+UJU^T)^{-1} x
        Nx  = x / self._nvec
        yNx = np.dot(y, Nx)
        for comp in self._components:
            idxs = comp['idxs']
            U    = comp['U']
            j    = comp['j']
            Dinv = 1.0 / self._nvec[idxs]

            Jinv = np.diag(1.0 / j)
            M    = Jinv + U.T.dot(Dinv[:,None] * U)
            invM = np.linalg.inv(M)

            kx = U.T.dot(Dinv * x[idxs])
            ky = U.T.dot(Dinv * y[idxs])
            yNx -= ky.dot(invM.dot(kx))
        return yNx

    def _solve_2D2(self, X, Z):
        # Z^T (N+UJU^T)^{-1} X
        # start with diagonal part:
        Dinv = 1.0 / self._nvec
        ZNX  = (Z * Dinv[:,None]).T.dot(X)
        for comp in self._components:
            idxs = comp['idxs']
            U    = comp['U']
            j    = comp['j']
            Dinv_c = Dinv[idxs][:,None]

            Jinv = np.diag(1.0 / j)
            M    = Jinv + U.T.dot(Dinv_c * U)
            invM = np.linalg.inv(M)

            # low-rank correction
            L = (Z[idxs,:] * Dinv_c).T.dot(U)         # shape (r × k)
            R =       U.T.dot(Dinv_c * X[idxs,:])    # shape (k × c)
            ZNX -= L.dot(invM.dot(R))
        return ZNX

    def _get_logdet(self):
        # log det of N + ∑ U_j j_j U_j^T
        ld = np.sum(np.log(self._nvec))
        for comp in self._components:
            idxs = comp['idxs']
            U    = comp['U']
            j    = comp['j']
            Dinv = 1.0/self._nvec[idxs]

            Jinv = np.diag(1.0/j)
            M    = Jinv + U.T.dot(Dinv[:,None] * U)
            sign, logdetM = np.linalg.slogdet(M)
            ld += np.sum(np.log(j)) + logdetM
        return ld

    def solve(self, other, left_array=None, logdet=False):
        # same dispatch logic as before, but calls our new helpers
        if other.ndim == 1:
            if left_array is None:
                ret = self._solve_D1(other)
            else:
                if left_array.ndim == 1:
                    ret = self._solve_1D1(other, left_array)
                else:
                    # y^T N^{-1} x path
                    ret = np.dot(left_array.T, self._solve_D1(other))
        elif other.ndim == 2:
            if left_array is None:
                raise NotImplementedError("ShermanMorrison does not implement _solve_D2")
            elif left_array.ndim == 2:
                ret = self._solve_2D2(other, left_array)
            elif left_array.ndim == 1:
                ret = np.dot(other.T, self._solve_D1(left_array))
            else:
                raise TypeError
        else:
            raise TypeError

        return (ret, self._get_logdet()) if logdet else ret

    def _sqrtsolve_D2(self, x):
        # initial divide‐out by sqrt(N)
        out = x / np.sqrt(self._nvec)[:,None]

        # for each component, do a full Cholesky of the small block
        for comp in self._components:
            idxs = comp['idxs']
            if len(idxs) == 1:
                # singletons already handled by N^{-1/2}
                continue

            D = self._nvec[idxs]
            A = np.diag(D)
            # add each rank-1 piece from this component
            for col, jv in enumerate(comp['j']):
                # where U[:,col] == 1
                mask = comp['U'][:,col].astype(bool)
                A[np.ix_(mask, mask)] += jv

            # now L L^T = A, solve L y = x[idxs]
            Lblock = sl.cholesky(A, lower=True)
            out[idxs,:] = sl.solve_triangular(Lblock, x[idxs,:], lower=True)

        return out

    def sqrtsolve(self, other, left_array=None):
        if other.ndim == 1:
            vec = other.reshape(-1,1)
            ret = self._sqrtsolve_D2(vec).ravel()
            if left_array is not None and left_array.ndim==1:
                return np.sum(left_array * ret)
            elif left_array is not None and left_array.ndim==2:
                return np.sum(left_array * ret[:,None], axis=0)
            else:
                return ret
        elif other.ndim == 2:
            if left_array is None:
                return self._sqrtsolve_D2(other)
            else:
                raise NotImplementedError
        else:
            raise TypeError


class FastShermanMorrisonRK:
    def __init__(self, jvec, slices, nvec):
        self._nvec = np.asarray(nvec)
        self._orig_idxs = [indices_from_slice(s) for s in slices]
        self._jvec      = np.asarray(jvec)
        self._has_sqrtsolve = True

        # Disjoint-set (union-find) initialization
        B = len(self._orig_idxs)
        parent = list(range(B))
        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i,j):
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[rj] = ri

        # Inverted index: map each position → list of blocks that contain it
        pos2blocks = defaultdict(list)
        for b, idx in enumerate(self._orig_idxs):
            for p in idx:
                pos2blocks[p].append(b)

        # Union all blocks that share any position
        for blocks in pos2blocks.values():
            first = blocks[0]
            for b in blocks[1:]:
                union(first,b)

        # Gather connected components of block-IDs
        comps = {}
        for b in range(B):
            root = find(b)
            comps.setdefault(root, []).append(b)

        # assemble per-component data
        self._components = []
        for comp_blocks in comps.values():
            all_idx = sorted({i for b in comp_blocks for i in self._orig_idxs[b]})
            pos_map = {idx:i for i, idx in enumerate(all_idx)}

            k = len(comp_blocks)
            D = len(all_idx)
            U = np.zeros((D, k))
            Uinds = np.zeros((k,2), dtype=int)
            comp_j = np.zeros(k, dtype=np.double)
            for col, b in enumerate(sorted(comp_blocks)):
                comp_j[col] = self._jvec[b]
                idxs = self._orig_idxs[b]
                Uinds[col,0] = pos_map[idxs[0]]
                Uinds[col,1] = pos_map[idxs[-1]] + 1
                for i in self._orig_idxs[b]:
                    U[pos_map[i], col] = 1.0

            self._components.append({
                'idxs' : np.array(all_idx, dtype=int),
                'U'    : U,
                'j'    : comp_j,
                'Uinds': Uinds
            })

    def _solve_D1(self, x):
        out = np.zeros_like(x)
        for comp in self._components:
            idx = comp['idxs']
            out[idx] = np.asarray(cfsm.cython_block_shermor_solve_D1_k(
                x[idx], self._nvec[idx], comp['j'], comp['U']
            ))

        return out

    def _solve_1D1(self, x, y):
        tot_logdet = 0.0
        tot_yNx    = 0.0
        for comp in self._components:
            idx = comp['idxs']
            logdet, yNx = cfsm.cython_block_shermor_solve_1D1_k(
                x[idx], y[idx], self._nvec[idx], comp['j'], comp['Uinds']
            )
            tot_logdet += logdet
            tot_yNx    += yNx
        return tot_logdet, tot_yNx

    def _solve_2D2(self, X, Z):
        total_logdet = 0.0
        ZNZ_acc = None
        for comp in self._components:
            idx = comp['idxs']
            subZ1 = X[idx,:]
            subZ2 = Z[idx,:]
            logdet, ZNZ = cfsm.cython_block_shermor_2D2_rankk(
                subZ1, subZ2, self._nvec[idx], comp['j'], comp['Uinds']
            )
            total_logdet += logdet
            if ZNZ_acc is None:
                ZNZ_acc = ZNZ.copy()
            else:
                ZNZ_acc += ZNZ
        return total_logdet, ZNZ_acc

    def solve(self, other, left_array=None, logdet=False):
        if other.ndim == 1:
            if left_array is None:
                ret = self._solve_D1(other)
            elif left_array.ndim == 1:
                logdet_val, yNx = self._solve_1D1(other, left_array)
                ret = yNx
            else:
                ret = left_array.T.dot(self._solve_D1(other))
        elif other.ndim == 2:
            if left_array is None:
                raise NotImplementedError
            else:
                logdet_val, ret = self._solve_2D2(other, left_array)
        else:
            raise TypeError
        return (ret, logdet_val) if logdet else ret

    def sqrtsolve(self, X):
        out = np.zeros_like(X)
        for comp in self._components:
            idx = comp['idxs']
            out[idx,:] = np.asarray(cfsm.cython_block_sqrtsolve_rankk(
                X[idx,:], self._nvec[idx], comp['j'], comp['Uinds']
            ))
        return out
