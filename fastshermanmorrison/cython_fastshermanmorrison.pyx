cimport numpy as cnp
import numpy as np
cnp.import_array()

from libc.math cimport log, sqrt, hypot
import cython
from scipy.linalg.cython_blas cimport dgemm, dger, dgemv


cdef public void dgemm_(char *transa, char *transb, int *m, int *n, int *k,
                          double *alpha, double *a, int *lda, double *b,
                          int *ldb, double *beta, double *c, int *ldc):
    """Public dgemm that can be used in the external C code"""
    dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)

cdef public void dgemv_(char *trans, int *m, int *n, double *alpha, double *a,
                        int *lda, double *x, int *incx, double *beta,
                        double *y, int *incy):
    """Public dgemv that can be used in the external C code"""
    dgemv(trans, m, n, alpha, a, lda, x, incx, beta, y, incy)

cdef public void dger_(int *m, int *n, double *alpha, double *x, int *incx,
                          double *y, int *incy, double *a, int *lda):
    """Public dger that can be used in the external C code"""
    dger(m, n, alpha, x, incx, y, incy, a, lda)

cdef extern from "fastshermanmorrison.c":
    void blas_block_shermor_2D_asym(
                int n_Z1_rows,
                int n_Z1_cols,
                int n_Z1_row_major,
                double *pd_Z1,
                int n_Z2_cols,
                int n_Z2_row_major,
                double *pd_Z2,
                double *pd_Nvec,
                int n_J_rows,
                double *pd_Jvec,
                int *pn_Uinds,
                double *pd_ZNZ,
                double *pd_Jldet
             )

@cython.boundscheck(False)
@cython.wraparound(False)
def cython_block_shermor_0D( \
        cnp.ndarray[cnp.double_t,ndim=1] r, \
        cnp.ndarray[cnp.double_t,ndim=1] Nvec, \
        cnp.ndarray[cnp.double_t,ndim=1] Jvec, \
        cnp.ndarray[cnp.int64_t,ndim=2] Uinds):
    """
    Sherman-Morrison block-inversion for Jitter (Cythonized)

    :param r:       The timing residuals, array (n)
    :param Nvec:    The white noise amplitude, array (n)
    :param Jvec:    The jitter amplitude, array (k)
    :param Uinds:   The start/finish indices for the jitter blocks (k x 2)

    For this version, the residuals need to be sorted properly so that all the
    blocks are continuous in memory. Here, there are n residuals, and k jitter
    parameters.
    """
    cdef unsigned int cc, ii, cols = len(Jvec)
    cdef double Jldet=0.0, ji, beta, nir, nisum
    cdef cnp.ndarray[cnp.double_t,ndim=1] ni = np.empty(len(r), 'd')
    cdef cnp.ndarray[cnp.double_t,ndim=1] Nx = r / Nvec

    ni = 1.0 / Nvec

    for cc in range(cols):
        if Jvec[cc] > 0.0:
            ji = 1.0 / Jvec[cc]

            nir = 0.0
            nisum = 0.0
            for ii in range(Uinds[cc,0],Uinds[cc,1]):
                nisum += ni[ii]
                nir += r[ii]*ni[ii]

            beta = 1.0 / (nisum + ji)

            for ii in range(Uinds[cc,0],Uinds[cc,1]):
                Nx[ii] -= beta * nir * ni[ii]

    return Nx

@cython.boundscheck(False)
@cython.wraparound(False)
def cython_idx_block_shermor_0D( \
        cnp.ndarray[cnp.double_t,ndim=1] r, \
        cnp.ndarray[cnp.double_t,ndim=1] Nvec, \
        cnp.ndarray[cnp.double_t,ndim=1] Jvec, \
        cnp.ndarray[cnp.int64_t,ndim=2] Uinds, \
        cnp.ndarray[cnp.int64_t,ndim=1] slc_isort):
    """
    Sherman-Morrison block-inversion for Jitter (Cythonized)

    :param r:           The timing residuals, array (n)
    :param Nvec:        The white noise amplitude, array (n)
    :param Jvec:        The jitter amplitude, array (k)
    :param Uinds:       The start/finish indices for the jitter blocks (k x 2)
    :param slc_isort:   The sorting of indices for which Uinds is valid

    For this version, the residuals need not be sorted Here, there are n
    residuals, and k jitter parameters.
    """
    cdef unsigned int cc, ii, idx, cols = len(Jvec)
    cdef double Jldet=0.0, ji, beta, nir, nisum
    cdef cnp.ndarray[cnp.double_t,ndim=1] ni = np.empty(len(r), 'd')
    cdef cnp.ndarray[cnp.double_t,ndim=1] Nx = r / Nvec

    ni = 1.0 / Nvec

    for cc in range(cols):
        if Jvec[cc] > 0.0:
            ji = 1.0 / Jvec[cc]

            nir = 0.0
            nisum = 0.0
            for ii in range(Uinds[cc,0],Uinds[cc,1]):
                idx = slc_isort[ii]
                nisum += ni[idx]
                nir += r[idx]*ni[idx]

            beta = 1.0 / (nisum + ji)

            for ii in range(Uinds[cc,0],Uinds[cc,1]):
                idx = slc_isort[ii]
                Nx[idx] -= beta * nir * ni[idx]

    return Nx

@cython.boundscheck(False)
@cython.wraparound(False)
def cython_block_shermor_0D_ld( \
        cnp.ndarray[cnp.double_t,ndim=1] r, \
        cnp.ndarray[cnp.double_t,ndim=1] Nvec, \
        cnp.ndarray[cnp.double_t,ndim=1] Jvec, \
        cnp.ndarray[cnp.int64_t,ndim=2] Uinds):
    """
    Sherman-Morrison block-inversion for Jitter (Cythonized)

    :param r:       The timing residuals, array (n)
    :param Nvec:    The white noise amplitude, array (n)
    :param Jvec:    The jitter amplitude, array (k)
    :param Uinds:   The start/finish indices for the jitter blocks (k x 2)

    For this version, the residuals need to be sorted properly so that all the
    blocks are continuous in memory. Here, there are n residuals, and k jitter
    parameters.
    """
    cdef unsigned int cc, ii, rows = len(r), cols = len(Jvec)
    cdef double Jldet=0.0, ji, beta, nir, nisum
    cdef cnp.ndarray[cnp.double_t,ndim=1] ni = np.empty(len(r), 'd')
    cdef cnp.ndarray[cnp.double_t,ndim=1] Nx = r / Nvec

    ni = 1.0 / Nvec

    for cc in range(rows):
        Jldet += log(Nvec[cc])

    for cc in range(cols):
        if Jvec[cc] > 0.0:
            ji = 1.0 / Jvec[cc]

            nir = 0.0
            nisum = 0.0
            for ii in range(Uinds[cc,0],Uinds[cc,1]):
                nisum += ni[ii]
                nir += r[ii]*ni[ii]

            beta = 1.0 / (nisum + ji)
            Jldet += log(Jvec[cc]) - log(beta)

            for ii in range(Uinds[cc,0],Uinds[cc,1]):
                Nx[ii] -= beta * nir * ni[ii]

    return Jldet, Nx

def python_block_shermor_1D(r, Nvec, Jvec, Uinds):
    """
    Sherman-Morrison block-inversion for Jitter

    :param r:       The timing residuals, array (n)
    :param Nvec:    The white noise amplitude, array (n)
    :param Jvec:    The jitter amplitude, array (k)
    :param Uinds:   The start/finish indices for the jitter blocks (k x 2)

    For this version, the residuals need not be sorted. Here, there are n
    residuals, and k jitter parameters.
    """
    ni = 1.0 / Nvec
    Jldet = np.einsum('i->', np.log(Nvec))
    xNx = np.dot(r, r * ni)

    for cc, jv in enumerate(Jvec):
        if jv > 0.0:
            rblock = r[Uinds[cc,0]:Uinds[cc,1]]
            niblock = ni[Uinds[cc,0]:Uinds[cc,1]]

            beta = 1.0 / (np.einsum('i->', niblock)+1.0/jv)
            xNx -= beta * np.dot(rblock, niblock)**2
            Jldet += np.log(jv) - np.log(beta)

    return Jldet, xNx

@cython.boundscheck(False)
@cython.wraparound(False)
def cython_block_shermor_1D( \
        cnp.ndarray[cnp.double_t,ndim=1] r, \
        cnp.ndarray[cnp.double_t,ndim=1] Nvec, \
        cnp.ndarray[cnp.double_t,ndim=1] Jvec, \
        cnp.ndarray[cnp.int64_t,ndim=2] Uinds):
    """
    Sherman-Morrison block-inversion for Jitter (Cythonized)

    :param r:       The timing residuals, array (n)
    :param Nvec:    The white noise amplitude, array (n)
    :param Jvec:    The jitter amplitude, array (k)
    :param Uinds:   The start/finish indices for the jitter blocks (k x 2)

    For this version, the residuals need to be sorted properly so that all the
    blocks are continuous in memory. Here, there are n residuals, and k jitter
    parameters.
    """
    cdef unsigned int cc, ii, rows = len(r), cols = len(Jvec)
    cdef double Jldet=0.0, ji, beta, xNx=0.0, nir, nisum
    cdef cnp.ndarray[cnp.double_t,ndim=1] ni = np.empty(rows, 'd')

    ni = 1.0 / Nvec

    for cc in range(rows):
        Jldet += log(Nvec[cc])
        xNx += r[cc]*r[cc]*ni[cc]

    for cc in range(cols):
        if Jvec[cc] > 0.0:
            ji = 1.0 / Jvec[cc]

            nir = 0.0
            nisum = 0.0
            for ii in range(Uinds[cc,0],Uinds[cc,1]):
                nisum += ni[ii]
                nir += r[ii]*ni[ii]

            beta = 1.0 / (nisum + ji)
            Jldet += log(Jvec[cc]) - log(beta)
            xNx -= beta * nir * nir

    return Jldet, xNx

@cython.boundscheck(False)
@cython.wraparound(False)
def cython_idx_block_shermor_1D( \
        cnp.ndarray[cnp.double_t,ndim=1] r, \
        cnp.ndarray[cnp.double_t,ndim=1] Nvec, \
        cnp.ndarray[cnp.double_t,ndim=1] Jvec, \
        cnp.ndarray[cnp.int64_t,ndim=2] Uinds, \
        cnp.ndarray[cnp.int64_t,ndim=1] slc_isort):
    """
    Sherman-Morrison block-inversion for Jitter (Cythonized)

    :param r:           The timing residuals, array (n)
    :param Nvec:        The white noise amplitude, array (n)
    :param Jvec:        The jitter amplitude, array (k)
    :param Uinds:       The start/finish indices for the jitter blocks (k x 2)
    :param slc_isort:   The sorting of indices for which Uinds is valid

    For this version, the residuals need not be sorted. Here, there are n
    residuals, and k jitter parameters.
    """
    cdef unsigned int cc, ii, idx, rows = len(r), cols = len(Jvec)
    cdef double Jldet=0.0, ji, beta, xNx=0.0, nir, nisum
    cdef cnp.ndarray[cnp.double_t,ndim=1] ni = np.empty(rows, 'd')

    ni = 1.0 / Nvec

    for cc in range(rows):
        Jldet += log(Nvec[cc])
        xNx += r[cc]*r[cc]*ni[cc]

    for cc in range(cols):
        if Jvec[cc] > 0.0:
            ji = 1.0 / Jvec[cc]

            nir = 0.0
            nisum = 0.0
            for ii in range(Uinds[cc,0],Uinds[cc,1]):
                idx = slc_isort[ii]
                nisum += ni[idx]
                nir += r[idx]*ni[idx]

            beta = 1.0 / (nisum + ji)
            Jldet += log(Jvec[cc]) - log(beta)
            xNx -= beta * nir * nir

    return Jldet, xNx

@cython.boundscheck(False)
@cython.wraparound(False)
def cython_block_shermor_1D1( \
        cnp.ndarray[cnp.double_t,ndim=1] x, \
        cnp.ndarray[cnp.double_t,ndim=1] y, \
        cnp.ndarray[cnp.double_t,ndim=1] Nvec, \
        cnp.ndarray[cnp.double_t,ndim=1] Jvec, \
        cnp.ndarray[cnp.int64_t,ndim=2] Uinds):
    """
    Sherman-Morrison block-inversion for Jitter (Cythonized)

    :param r:       The timing residuals, array (n)
    :param Nvec:    The white noise amplitude, array (n)
    :param Jvec:    The jitter amplitude, array (k)
    :param Uinds:   The start/finish indices for the jitter blocks (k x 2)

    For this version, the residuals need to be sorted properly so that all the
    blocks are continuous in memory. Here, there are n residuals, and k jitter
    parameters.
    """
    cdef unsigned int cc, ii, rows = len(x), cols = len(Jvec)
    cdef double Jldet=0.0, ji, beta, yNx=0.0, nix, niy, nisum
    cdef cnp.ndarray[cnp.double_t,ndim=1] ni = np.empty(rows, 'd')

    ni = 1.0 / Nvec

    for cc in range(rows):
        Jldet += log(Nvec[cc])
        yNx += y[cc]*x[cc]*ni[cc]

    for cc in range(cols):
        if Jvec[cc] > 0.0:
            ji = 1.0 / Jvec[cc]

            nix = 0.0
            niy = 0.0
            nisum = 0.0
            for ii in range(Uinds[cc,0],Uinds[cc,1]):
                nisum += ni[ii]
                nix += x[ii]*ni[ii]
                niy += y[ii]*ni[ii]

            beta = 1.0 / (nisum + ji)
            Jldet += log(Jvec[cc]) - log(beta)
            yNx -= beta * nix * niy

    return Jldet, yNx

@cython.boundscheck(False)
@cython.wraparound(False)
def cython_idx_block_shermor_1D1( \
        cnp.ndarray[cnp.double_t,ndim=1] x, \
        cnp.ndarray[cnp.double_t,ndim=1] y, \
        cnp.ndarray[cnp.double_t,ndim=1] Nvec, \
        cnp.ndarray[cnp.double_t,ndim=1] Jvec, \
        cnp.ndarray[cnp.int64_t,ndim=2] Uinds, \
        cnp.ndarray[cnp.int64_t,ndim=1] slc_isort):
    """
    Sherman-Morrison block-inversion for Jitter (Cythonized)

    :param r:           The timing residuals, array (n)
    :param Nvec:        The white noise amplitude, array (n)
    :param Jvec:        The jitter amplitude, array (k)
    :param Uinds:       The start/finish indices for the jitter blocks (k x 2)
    :param slc_isort:   The sorting of indices for which Uinds is valid

    For this version, the residuals need not be sorted. Here, there are n
    residuals, and k jitter parameters.
    """
    cdef unsigned int cc, ii, idx, rows = len(x), cols = len(Jvec)
    cdef double Jldet=0.0, ji, beta, yNx=0.0, nix, niy, nisum
    cdef cnp.ndarray[cnp.double_t,ndim=1] ni = np.empty(rows, 'd')

    ni = 1.0 / Nvec

    for cc in range(rows):
        Jldet += log(Nvec[cc])
        yNx += y[cc]*x[cc]*ni[cc]

    for cc in range(cols):
        if Jvec[cc] > 0.0:
            ji = 1.0 / Jvec[cc]

            nix = 0.0
            niy = 0.0
            nisum = 0.0
            for ii in range(Uinds[cc,0],Uinds[cc,1]):
                idx = slc_isort[ii]
                nisum += ni[idx]
                nix += x[idx]*ni[idx]
                niy += y[idx]*ni[idx]

            beta = 1.0 / (nisum + ji)
            Jldet += log(Jvec[cc]) - log(beta)
            yNx -= beta * nix * niy

    return Jldet, yNx

def python_block_shermor_2D(Z, Nvec, Jvec, Uinds):
    """
    Sherman-Morrison block-inversion for Jitter, ZNiZ

    :param Z:       The design matrix, array (n x m)
    :param Nvec:    The white noise amplitude, array (n)
    :param Jvec:    The jitter amplitude, array (k)
    :param Uinds:   The start/finish indices for the jitter blocks (k x 2)

    For this version, the residuals need to be sorted properly so that all the
    blocks are continuous in memory. Here, there are n residuals, and k jitter
    parameters.
    
    N = D + U*J*U.T
    calculate: log(det(N)), Z.T * N^-1 * Z
    """
    ni = 1.0 / Nvec
    Jldet = np.einsum('i->', np.log(Nvec))
    zNz = np.dot(Z.T*ni, Z)

    for cc, jv in enumerate(Jvec):
        if jv > 0.0:
            Zblock = Z[Uinds[cc,0]:Uinds[cc,1], :]
            niblock = ni[Uinds[cc,0]:Uinds[cc,1]]

            beta = 1.0 / (np.einsum('i->', niblock)+1.0/jv)
            zn = np.dot(niblock, Zblock)
            zNz -= beta * np.outer(zn.T, zn)
            Jldet += np.log(jv) - np.log(beta)

    return Jldet, zNz

@cython.boundscheck(False)
@cython.wraparound(False)
def cython_block_shermor_2D( \
        cnp.ndarray[cnp.double_t,ndim=2] Z, \
        cnp.ndarray[cnp.double_t,ndim=1] Nvec, \
        cnp.ndarray[cnp.double_t,ndim=1] Jvec, \
        cnp.ndarray[cnp.int64_t,ndim=2] Uinds):
    """
    Sherman-Morrison block-inversion for Jitter (Cythonized)

    :param Z:       The design matrix, array (n x m)
    :param Nvec:    The white noise amplitude, array (n)
    :param Jvec:    The jitter amplitude, array (k)
    :param Uinds:   The start/finish indices for the jitter blocks (k x 2)

    For this version, the residuals need to be sorted properly so that all the
    blocks are continuous in memory. Here, there are n residuals, and k jitter
    parameters.

    N = D + U*J*U.T
    calculate: log(det(N)), Z.T * N^-1 * Z
    """
    cdef unsigned int cc, ii, rows = len(Nvec), cols = len(Jvec)
    cdef double Jldet=0.0, ji, beta, nir, nisum
    cdef cnp.ndarray[cnp.double_t,ndim=1] ni = np.empty(len(Nvec), 'd')
    cdef cnp.ndarray[cnp.double_t,ndim=2] zNz

    ni = 1.0 / Nvec
    zNz = np.dot(Z.T*ni, Z)

    for cc in range(rows):
        Jldet += log(Nvec[cc])

    for cc in range(cols):
        if Jvec[cc] > 0.0:
            Zblock = Z[Uinds[cc,0]:Uinds[cc,1], :]
            niblock = ni[Uinds[cc,0]:Uinds[cc,1]]

            nisum = 0.0
            for ii in range(len(niblock)):
                nisum += niblock[ii]

            beta = 1.0 / (nisum+1.0/Jvec[cc])
            Jldet += log(Jvec[cc]) - log(beta)
            zn = np.dot(niblock, Zblock)
            zNz -= beta * np.outer(zn.T, zn)

    return Jldet, zNz

def python_block_shermor_2D_asymm(Z1, Z2, Nvec, Jvec, Uinds):
    """
    Sherman-Morrison block-inversion for Jitter, ZNiZ

    :param Z:       The design matrix, array (n x m)
    :param Z2:      The second design matrix, array (n x m2)
    :param Nvec:    The white noise amplitude, array (n)
    :param Jvec:    The jitter amplitude, array (k)
    :param Uinds:   The start/finish indices for the jitter blocks (k x 2)

    For this version, the residuals need to be sorted properly so that all the
    blocks are continuous in memory. Here, there are n residuals, and k jitter
    parameters.

    N = D + U*J*U.T
    calculate: log(det(N)), Z.T * N^-1 * Z
    """
    ni = 1.0 / Nvec
    Jldet = np.einsum('i->', np.log(Nvec))
    zNz = np.dot(Z1.T*ni, Z2)

    for cc, jv in enumerate(Jvec):
        if jv > 0.0:
            Zblock1 = Z1[Uinds[cc,0]:Uinds[cc,1], :]
            Zblock2 = Z2[Uinds[cc,0]:Uinds[cc,1], :]
            niblock = ni[Uinds[cc,0]:Uinds[cc,1]]

            beta = 1.0 / (np.einsum('i->', niblock)+1.0/jv)
            zn1 = np.dot(niblock, Zblock1)
            zn2 = np.dot(niblock, Zblock2)
            zNz -= beta * np.outer(zn1.T, zn2)
            Jldet += np.log(jv) - np.log(beta)

    return Jldet, zNz

@cython.boundscheck(False)
@cython.wraparound(False)
def cython_block_shermor_2D_asymm(
        cnp.ndarray[cnp.double_t,ndim=2] Z1,
        cnp.ndarray[cnp.double_t,ndim=2] Z2,
        cnp.ndarray[cnp.double_t,ndim=1] Nvec,
        cnp.ndarray[cnp.double_t,ndim=1] Jvec,
        cnp.ndarray[cnp.int64_t,ndim=2] Uinds):
    """
    Sherman-Morrison block-inversion for Jitter, ZNiZ

    :param Z:       The design matrix, array (n x m)
    :param Z2:      The second design matrix, array (n x m2)
    :param Nvec:    The white noise amplitude, array (n)
    :param Jvec:    The jitter amplitude, array (k)
    :param Uinds:   The start/finish indices for the jitter blocks (k x 2)

    For this version, the residuals need to be sorted properly so that all the
    blocks are continuous in memory. Here, there are n residuals, and k jitter
    parameters.

    N = D + U*J*U.T
    calculate: log(det(N)), Z.T * N^-1 * Z
    """
    cdef unsigned int cc, ii, rows = len(Nvec), cols = len(Jvec)
    cdef double Jldet=0.0, ji, beta, nir, nisum
    cdef cnp.ndarray[cnp.double_t,ndim=1] ni = np.empty(len(Nvec), 'd')
    cdef cnp.ndarray[cnp.double_t,ndim=2] zNz

    print("WARNING: cython_block_shermor_2D_asymm is deprecated.")
    print("         use cython_blas_block_shermor_2D_asymm")

    ni = 1.0 / Nvec
    for cc in range(rows):
        Jldet += log(Nvec[cc])
    zNz = np.dot(Z1.T*ni, Z2)

    for cc in range(cols):
        if Jvec[cc] > 0.0:
            Zblock1 = Z1[Uinds[cc,0]:Uinds[cc,1], :]
            Zblock2 = Z2[Uinds[cc,0]:Uinds[cc,1], :]
            niblock = ni[Uinds[cc,0]:Uinds[cc,1]]

            nisum = 0.0
            for ii in range(len(niblock)):
                nisum += niblock[ii]

            beta = 1.0 / (nisum+1.0/Jvec[cc])
            zn1 = np.dot(niblock, Zblock1)
            zn2 = np.dot(niblock, Zblock2)

            zNz -= beta * np.outer(zn1.T, zn2)
            Jldet += log(Jvec[cc]) - log(beta)

    return Jldet, zNz


def python_block_sqrtsolve_rank1(X, Nvec, Jvec, Uinds):
    """
    Block‑wise solve L_block^{-1} X_block using true triangular rank‑1 update.

    Parameters
    ----------
    X      : ndarray, shape (n, ℓ)
    Nvec   : ndarray, shape (n,)
    Jvec   : ndarray, shape (k,)
    Uinds  : array-like of shape (k, 2)

    Returns
    -------
    Lix : ndarray, shape (n, ℓ)
    """
    Lix = np.zeros_like(X)
    for idx, jv in zip(Uinds, Jvec):
        start, end = idx
        Xb = X[start:end, :].copy()
        d  = Nvec[start:end]
        k, _ = Xb.shape

        L = np.diag(np.sqrt(d))
        w = np.sqrt(jv) * np.ones(k)

        for i in range(k):
            r = np.hypot(L[i,i], w[i])
            c = r / L[i,i]
            s = w[i] / L[i,i]
            L[i,i] = r
            if i+1 < k:
                Li1 = L[i+1:, i]
                wi1 = w[i+1:]
                L[i+1:, i] = (Li1 + s * wi1) / c
                w[i+1:] = c * wi1 - s * L[i+1:, i]

        Yb = Xb.copy()
        for i in range(k):
            Yb[i, :] /= L[i, i]
            if i+1 < k:
                Yb[i+1:, :] -= np.outer(L[i+1:, i], Yb[i, :])

        Lix[start:end, :] = Yb
    return Lix


def python_idx_sqrtsolve_rank1(X, Nvec, Jvec, Uinds, slc_isort):
    """
    Indexed version: applies rank‑1 block solve to unsorted rows.
    """
    Lix = np.zeros_like(X)
    for bi, jv in enumerate(Jvec):
        idx0, idx1 = Uinds[bi]
        k = idx1 - idx0
        Xb = np.zeros((k, X.shape[1]))
        d  = np.zeros(k)
        for i in range(k):
            pos = slc_isort[idx0 + i]
            Xb[i, :] = X[pos, :]
            d[i]      = Nvec[pos]

        L = np.diag(np.sqrt(d))
        w = np.sqrt(jv) * np.ones(k)
        for i in range(k):
            r = np.hypot(L[i,i], w[i])
            c = r / L[i,i]
            s = w[i] / L[i,i]
            L[i,i] = r
            if i+1 < k:
                Li1 = L[i+1:, i]
                wi1 = w[i+1:]
                L[i+1:, i] = (Li1 + s * wi1) / c
                w[i+1:] = c * wi1 - s * L[i+1:, i]

        Yb = Xb.copy()
        for i in range(k):
            Yb[i, :] /= L[i, i]
            if i+1 < k:
                Yb[i+1:, :] -= np.outer(L[i+1:, i], Yb[i, :])
        for i in range(k):
            pos = slc_isort[idx0 + i]
            Lix[pos, :] = Yb[i, :]

    return Lix


@cython.boundscheck(False)
@cython.wraparound(False)
def cython_block_sqrtsolve_rank1(
        cnp.ndarray[cnp.double_t, ndim=2] X,
        cnp.ndarray[cnp.double_t, ndim=1] Nvec,
        cnp.ndarray[cnp.double_t, ndim=1] Jvec,
        cnp.ndarray[cnp.int64_t, ndim=2] Uinds
    ):
    """
    Block‑wise forward solve L_block^{-1} X_block via true rank‑1 update.
    """
    cdef int n = X.shape[0]
    cdef int l = X.shape[1]
    cdef int k = Jvec.shape[0]
    cdef int bi, i, j, idx0, idx1, blk_len
    cdef double ri, ci, si

    cdef cnp.ndarray[cnp.double_t, ndim=2] Lix = np.zeros_like(X)
    cdef double[:, :] Xb
    cdef double[:, :] Yb
    cdef double[:, :] L
    cdef double[:] w

    for bi in range(k):
        idx0 = Uinds[bi, 0]
        idx1 = Uinds[bi, 1]
        blk_len = idx1 - idx0

        Xb = X[idx0:idx1, :]
        Yb = np.empty((blk_len, l), dtype=np.double)
        L = np.zeros((blk_len, blk_len), dtype=np.double)
        w = np.empty(blk_len, dtype=np.double)

        for i in range(blk_len):
            L[i, i] = sqrt(Nvec[idx0 + i])
            w[i] = sqrt(Jvec[bi])

        for i in range(blk_len):
            ri = hypot(L[i, i], w[i])
            ci = ri / L[i, i]
            si = w[i] / L[i, i]
            L[i, i] = ri
            for j in range(i+1, blk_len):
                L[j, i] = (L[j, i] + si * w[j]) / ci
                w[j] = ci * w[j] - si * L[j, i]

        for j in range(l):
            for i in range(blk_len):
                Yb[i, j] = Xb[i, j]
            for i in range(blk_len):
                Yb[i, j] /= L[i, i]
                for j2 in range(i+1, blk_len):
                    Yb[j2, j] -= L[j2, i] * Yb[i, j]

        Lix[idx0:idx1, :] = Yb

    return Lix


@cython.boundscheck(False)
@cython.wraparound(False)
def cython_idx_sqrtsolve_rank1(
        cnp.ndarray[cnp.double_t, ndim=2] X,
        cnp.ndarray[cnp.double_t, ndim=1] Nvec,
        cnp.ndarray[cnp.double_t, ndim=1] Jvec,
        cnp.ndarray[cnp.int64_t, ndim=2] Uinds,
        cnp.ndarray[cnp.int64_t, ndim=1] slc_isort
    ):
    """
    Indexed forward solve L_block^{-1} X_block via true rank‑1 update.
    """
    cdef int n = X.shape[0]
    cdef int l = X.shape[1]
    cdef int k = Jvec.shape[0]
    cdef int bi, i, j, idx0, idx1, blk_len, pos
    cdef double ri, ci, si

    cdef cnp.ndarray[cnp.double_t, ndim=2] Lix = np.zeros_like(X)
    cdef double[:, :] Xb
    cdef double[:, :] Yb
    cdef double[:, :] L
    cdef double[:] w

    for bi in range(k):
        idx0 = Uinds[bi, 0]
        idx1 = Uinds[bi, 1]
        blk_len = idx1 - idx0

        Xb = np.empty((blk_len, l), dtype=np.double)
        Yb = np.empty((blk_len, l), dtype=np.double)
        L = np.zeros((blk_len, blk_len), dtype=np.double)
        w = np.empty(blk_len, dtype=np.double)

        for i in range(blk_len):
            pos = slc_isort[idx0 + i]
            for j in range(l):
                Xb[i, j] = X[pos, j]
            L[i, i] = sqrt(Nvec[pos])
            w[i] = sqrt(Jvec[bi])

        for i in range(blk_len):
            ri = hypot(L[i, i], w[i])
            ci = ri / L[i, i]
            si = w[i] / L[i, i]
            L[i, i] = ri
            for j in range(i+1, blk_len):
                L[j, i] = (L[j, i] + si * w[j]) / ci
                w[j] = ci * w[j] - si * L[j, i]

        for j in range(l):
            for i in range(blk_len):
                Yb[i, j] = Xb[i, j]
            for i in range(blk_len):
                Yb[i, j] /= L[i, i]
                for j2 in range(i+1, blk_len):
                    Yb[j2, j] -= L[j2, i] * Yb[i, j]

        for i in range(blk_len):
            pos = slc_isort[idx0 + i]
            for j in range(l):
                Lix[pos, j] = Yb[i, j]

    return Lix


def python_draw_ecor(r, Nvec, Jvec, Uinds):
    """
    Given Jvec, draw new epoch-averaged residuals

    :param r:       The timing residuals, array (n)
    :param Nvec:    The white noise amplitude, array (n)
    :param Jvec:    The jitter amplitude, array (k)
    :param Uinds:   The start/finish indices for the jitter blocks (k x 2)

    For this version, the residuals need to be sorted properly so that all the
    blocks are continuous in memory. Here, there are n residuals, and k jitter
    parameters.
    
    N = D + U*J*U.T
    calculate: Norm(0, sqrt(J)) + (U^T * D^{-1} * U)^{-1}U.T D^{-1} r
    """
    
    rv = np.random.randn(len(Jvec)) * np.sqrt(Jvec)
    ni = 1.0 / Nvec

    for cc in range(len(Jvec)):
        rblock = r[Uinds[cc,0]:Uinds[cc,1]]
        niblock = ni[Uinds[cc,0]:Uinds[cc,1]]
        beta = 1.0 / np.einsum('i->', niblock)

        rv[cc] += beta * np.dot(rblock, niblock)

    return rv

@cython.boundscheck(False)
@cython.wraparound(False)
def cython_draw_ecor( \
        cnp.ndarray[cnp.double_t,ndim=1] r, \
        cnp.ndarray[cnp.double_t,ndim=1] Nvec, \
        cnp.ndarray[cnp.double_t,ndim=1] Jvec, \
        cnp.ndarray[cnp.int64_t,ndim=2] Uinds):
    """
    Given Jvec, draw new epoch-averaged residuals

    :param r:       The timing residuals, array (n)
    :param Nvec:    The white noise amplitude, array (n)
    :param Jvec:    The jitter amplitude, array (k)
    :param Uinds:   The start/finish indices for the jitter blocks (k x 2)

    For this version, the residuals need to be sorted properly so that all the
    blocks are continuous in memory. Here, there are n residuals, and k jitter
    parameters.

    N = D + U*J*U.T
    calculate: Norm(0, sqrt(J)) + (U^T * D^{-1} * U)^{-1}U.T D^{-1} r
    """
    cdef unsigned int cc, ii, rows = len(r), cols = len(Jvec)
    cdef double ji, nir, nisum
    cdef cnp.ndarray[cnp.double_t,ndim=1] ni = np.empty(rows, 'd')
    cdef cnp.ndarray[cnp.double_t,ndim=1] rv = np.random.randn(cols)

    for cc in range(cols):
        rv[cc] *= sqrt(Jvec[cc])

    ni = 1.0 / Nvec

    for cc in range(cols):
        ji = 1.0 / Jvec[cc]

        nir = 0.0
        nisum = 0.0
        for ii in range(Uinds[cc,0],Uinds[cc,1]):
            nisum += ni[ii]
            nir += r[ii]*ni[ii]

        rv[cc] += nir / nisum

    return rv

@cython.boundscheck(False)
@cython.wraparound(False)
def cython_shermor_draw_ecor( \
        cnp.ndarray[cnp.double_t,ndim=1] r, \
        cnp.ndarray[cnp.double_t,ndim=1] Nvec, \
        cnp.ndarray[cnp.double_t,ndim=1] Jvec, \
        cnp.ndarray[cnp.int64_t,ndim=2] Uinds):
    """
    Do both the Sherman-Morrison block-inversion for Jitter,
    and the draw of the ecor parameters together (Cythonized)

    :param r:       The timing residuals, array (n)
    :param Nvec:    The white noise amplitude, array (n)
    :param Jvec:    The jitter amplitude, array (k)
    :param Uinds:   The start/finish indices for the jitter blocks (k x 2)

    For this version, the residuals need to be sorted properly so that all the
    blocks are continuous in memory. Here, there are n residuals, and k jitter
    parameters.

    N = D + U*J*U.T
    calculate: r.T * N^-1 * r, log(det(N)), Norm(0, sqrt(J)) + (U^T * D^{-1} * U)^{-1}U.T D^{-1} r
    """
    cdef unsigned int cc, ii, rows = len(r), cols = len(Jvec)
    cdef double Jldet=0.0, ji, beta, xNx=0.0, nir, nisum
    cdef cnp.ndarray[cnp.double_t,ndim=1] ni = np.empty(rows, 'd')
    cdef cnp.ndarray[cnp.double_t,ndim=1] rv = np.random.randn(cols)

    ni = 1.0 / Nvec

    for cc in range(cols):
        rv[cc] *= sqrt(Jvec[cc])

    for cc in range(rows):
        Jldet += log(Nvec[cc])
        xNx += r[cc]*r[cc]*ni[cc]

    for cc in range(cols):
        nir = 0.0
        nisum = 0.0
        for ii in range(Uinds[cc,0],Uinds[cc,1]):
            nisum += ni[ii]
            nir += r[ii]*ni[ii]

        rv[cc] += nir / nisum

        if Jvec[cc] > 0.0:
            ji = 1.0 / Jvec[cc]

            beta = 1.0 / (nisum + ji)
            Jldet += log(Jvec[cc]) - log(beta)
            xNx -= beta * nir * nir

    return Jldet, xNx, rv

@cython.boundscheck(False)
@cython.wraparound(False)
def cython_update_ea_residuals( \
        cnp.ndarray[cnp.double_t,ndim=1] gibbsresiduals, \
        cnp.ndarray[cnp.double_t,ndim=1] gibbssubresiduals, \
        cnp.ndarray[cnp.double_t,ndim=1] eat, \
        cnp.ndarray[cnp.int64_t,ndim=2] Uinds):
    """
    Given epoch-averaged residuals, update the residuals, and the subtracted
    residuals, so that these can be further processed by the other conditional
    probability density functions.

    :param gibbsresiduals:      The timing residuals, array (n)
    :param gibbssubresiduals:   The white noise amplitude, array (n)
    :param eat:                 epoch averaged residuals (k)
    :param Uinds:               The start/finish indices for the jitter blocks
                                (k x 2)

    """
    cdef unsigned int k = Uinds.shape[0], ii, cc

    for cc in range(Uinds.shape[0]):
        for ii in range(Uinds[cc,0],Uinds[cc,1]):
            gibbssubresiduals[ii] += eat[cc]
            gibbsresiduals[ii] -= eat[cc]

    return gibbsresiduals, gibbssubresiduals


@cython.boundscheck(False)
@cython.wraparound(False)
def cython_Uj(cnp.ndarray[cnp.double_t,ndim=1] j, \
        cnp.ndarray[cnp.int64_t,ndim=2] Uinds, nobs):
    """
    Given epoch-averaged residuals (j), get the residuals.
    Used in 'updateDetSources'

    :param j:                   epoch averaged residuals (k)
    :param Uinds:               The start/finish indices for the jitter blocks
                                (k x 2)
    :param nobs:                Number of observations (length return vector)

    """
    cdef unsigned int k = Uinds.shape[0], ii, cc
    cdef cnp.ndarray[cnp.double_t,ndim=1] Uj = np.zeros(nobs, 'd')

    for cc in range(k):
        for ii in range(Uinds[cc,0],Uinds[cc,1]):
            Uj[ii] += j[cc]

    return Uj

@cython.boundscheck(False)
@cython.wraparound(False)
def cython_UTx(cnp.ndarray[cnp.double_t,ndim=1] x, \
        cnp.ndarray[cnp.int64_t,ndim=2] Uinds):
    """
    Given residuals (x), get np.dot(U.T, x)
    Used in 'updateDetSources'

    :param j:                   epoch averaged residuals (k)
    :param Uinds:               The start/finish indices for the jitter blocks
                                (k x 2)

    """
    cdef unsigned int k = Uinds.shape[0], ii, cc
    cdef cnp.ndarray[cnp.double_t,ndim=1] UTx = np.zeros(k, 'd')

    for cc in range(k):
        for ii in range(Uinds[cc,0],Uinds[cc,1]):
            UTx[cc] += x[ii]

    return UTx

@cython.boundscheck(False)
@cython.wraparound(False)
def cython_logdet_dN( \
        cnp.ndarray[cnp.double_t,ndim=1] Nvec, \
        cnp.ndarray[cnp.double_t,ndim=1] Jvec, \
        cnp.ndarray[cnp.double_t,ndim=1] dNvec, \
        cnp.ndarray[cnp.int64_t,ndim=2] Uinds):
    """
    Sherman-Morrison block-inversion for Jitter (Cythonized)

    Calculates Trace(N^{-1} dN/dNp), where:
        - N^{-1} is the ecorr-include N inverse
        - dN/dNp is the diagonal derivate of N wrt Np

    :param Nvec:    The white noise amplitude, array (n)
    :param Jvec:    The jitter amplitude, array (k)
    :param dNvec:   The white noise derivative, array (n)
    :param Uinds:   The start/finish indices for the jitter blocks (k x 2)

    For this version, the residuals need to be sorted properly so that all the
    blocks are continuous in memory. Here, there are n residuals, and k jitter
    parameters.
    """
    cdef unsigned int cc, ii, rows = len(Nvec), cols = len(Jvec)
    cdef double tr=0.0, ji, nisum, Nnisum
    cdef cnp.ndarray[cnp.double_t,ndim=1] ni = np.empty(rows, 'd')
    cdef cnp.ndarray[cnp.double_t,ndim=1] Nni = np.empty(rows, 'd')

    ni = 1.0 / Nvec
    Nni = dNvec / Nvec**2

    for cc in range(rows):
        tr += dNvec[cc] * ni[cc]

    for cc in range(cols):
        if Jvec[cc] > 0.0:
            ji = 1.0 / Jvec[cc]

            nisum = 0.0
            Nnisum = 0.0
            for ii in range(Uinds[cc,0],Uinds[cc,1]):
                nisum += ni[ii]
                Nnisum += Nni[ii]

            tr -= Nnisum / (nisum + ji)

    return tr

@cython.boundscheck(False)
@cython.wraparound(False)
def cython_logdet_dJ( \
        cnp.ndarray[cnp.double_t,ndim=1] Nvec, \
        cnp.ndarray[cnp.double_t,ndim=1] Jvec, \
        cnp.ndarray[cnp.double_t,ndim=1] dJvec, \
        cnp.ndarray[cnp.int64_t,ndim=2] Uinds):
    """
    Sherman-Morrison block-inversion for Jitter (Cythonized)

    Calculates Trace(N^{-1} dN/dJp), where:
        - N^{-1} is the ecorr-include N inverse
        - dN/dJp = U dJ/dJp U^{T}, with dJ/dJp the diagnal derivative of J wrt
          Jp

    :param Nvec:    The white noise amplitude, array (n)
    :param Jvec:    The jitter amplitude, array (k)
    :param dJvec:   The jitter derivative, array (k)
    :param Uinds:   The start/finish indices for the jitter blocks (k x 2)

    For this version, the residuals need to be sorted properly so that all the
    blocks are continuous in memory. Here, there are n residuals, and k jitter
    parameters.
    """
    cdef unsigned int cc, ii, rows = len(Nvec), cols = len(Jvec)
    cdef double dJldet=0.0, ji, beta, nisum
    cdef cnp.ndarray[cnp.double_t,ndim=1] ni = np.empty(rows, 'd')

    ni = 1.0 / Nvec

    for cc in range(cols):
        if Jvec[cc] > 0.0:
            ji = 1.0 / Jvec[cc]

            nisum = 0.0
            for ii in range(Uinds[cc,0],Uinds[cc,1]):
                nisum += ni[ii]

            beta = 1.0 / (nisum + ji)

            dJldet += dJvec[cc]*(nisum - beta*nisum**2)

    return dJldet

@cython.boundscheck(False)
@cython.wraparound(False)
def cython_logdet_dN_dN( \
        cnp.ndarray[cnp.double_t,ndim=1] Nvec, \
        cnp.ndarray[cnp.double_t,ndim=1] Jvec, \
        cnp.ndarray[cnp.double_t,ndim=1] dNvec1, \
        cnp.ndarray[cnp.double_t,ndim=1] dNvec2, \
        cnp.ndarray[cnp.int64_t,ndim=2] Uinds):
    """
    Sherman-Morrison block-inversion for Jitter (Cythonized)

    Calculates Trace(N^{-1} dN/dNp1 N^{-1} dN/dNp2), where:
        - N^{-1} is the ecorr-include N inverse
        - dN/dNpx is the diagonal derivate of N wrt Npx

    :param Nvec:    The white noise amplitude, array (n)
    :param Jvec:    The jitter amplitude, array (k)
    :param dNvec1:  The white noise derivative, array (n)
    :param dNvec2:  The white noise derivative, array (n)
    :param Uinds:   The start/finish indices for the jitter blocks (k x 2)

    For this version, the residuals need to be sorted properly so that all the
    blocks are continuous in memory. Here, there are n residuals, and k jitter
    parameters.
    """
    cdef unsigned int cc, ii, rows = len(Nvec), cols = len(Jvec)
    cdef double tr=0.0, ji, nisum, Nnisum1, Nnisum2, NniNnisum, beta
    cdef cnp.ndarray[cnp.double_t,ndim=1] ni = np.empty(rows, 'd')
    cdef cnp.ndarray[cnp.double_t,ndim=1] Nni1 = np.empty(rows, 'd')
    cdef cnp.ndarray[cnp.double_t,ndim=1] Nni2 = np.empty(rows, 'd')

    ni = 1.0 / Nvec
    Nni1 = dNvec1 / Nvec**2
    Nni2 = dNvec2 / Nvec**2

    for cc in range(rows):
        tr += dNvec1[cc] * dNvec2[cc] * ni[cc]**2

    for cc in range(cols):
        if Jvec[cc] > 0.0:
            ji = 1.0 / Jvec[cc]

            nisum = 0.0
            Nnisum1 = 0.0
            Nnisum2 = 0.0
            NniNnisum = 0.0
            for ii in range(Uinds[cc,0],Uinds[cc,1]):
                nisum += ni[ii]
                Nnisum1 += Nni1[ii]
                Nnisum2 += Nni2[ii]
                NniNnisum += Nni1[ii]*Nni2[ii]*Nvec[ii]

            beta = 1.0 / (nisum + ji)

            tr += Nnisum1 * Nnisum2 * beta**2
            tr -= 2 * NniNnisum * beta

    return tr

@cython.boundscheck(False)
@cython.wraparound(False)
def cython_logdet_dN_dJ( \
        cnp.ndarray[cnp.double_t,ndim=1] Nvec, \
        cnp.ndarray[cnp.double_t,ndim=1] Jvec, \
        cnp.ndarray[cnp.double_t,ndim=1] dNvec, \
        cnp.ndarray[cnp.double_t,ndim=1] dJvec, \
        cnp.ndarray[cnp.int64_t,ndim=2] Uinds):
    """
    Sherman-Morrison block-inversion for Jitter (Cythonized)

    Calculates Trace(N^{-1} dN/dNp N^{-1} dN/dJp), where:
        - N^{-1} is the ecorr-include N inverse
        - dN/dNp is the diagonal derivate of N wrt Np
        - dN/dJp = U dJ/dJp U^{T}, with dJ/dJp the diagnal derivative of J wrt
          Jp

    :param Nvec:    The white noise amplitude, array (n)
    :param Jvec:    The jitter amplitude, array (k)
    :param dNvec:   The white noise derivative, array (n)
    :param dJvec:   The white noise ecor derivative, array (k)
    :param Uinds:   The start/finish indices for the jitter blocks (k x 2)

    For this version, the residuals need to be sorted properly so that all the
    blocks are continuous in memory. Here, there are n residuals, and k jitter
    parameters.
    """
    cdef unsigned int cc, ii, rows = len(Nvec), cols = len(Jvec)
    cdef double tr=0.0, ji, nisum, Nnisum, beta
    cdef cnp.ndarray[cnp.double_t,ndim=1] ni = np.empty(rows, 'd')
    cdef cnp.ndarray[cnp.double_t,ndim=1] Nni = np.empty(rows, 'd')

    ni = 1.0 / Nvec
    Nni = dNvec / Nvec**2

    for cc in range(cols):
        if Jvec[cc] > 0.0:
            ji = 1.0 / Jvec[cc]

            nisum = 0.0
            Nnisum = 0.0
            for ii in range(Uinds[cc,0],Uinds[cc,1]):
                nisum += ni[ii]
                Nnisum += Nni[ii]

            beta = 1.0 / (nisum + ji)

            tr += Nnisum * dJvec[cc]
            tr -= 2 * nisum * dJvec[cc] * Nnisum * beta
            tr += Nnisum * nisum**2 * dJvec[cc] *beta**2

    return tr

@cython.boundscheck(False)
@cython.wraparound(False)
def cython_logdet_dJ_dJ( \
        cnp.ndarray[cnp.double_t,ndim=1] Nvec, \
        cnp.ndarray[cnp.double_t,ndim=1] Jvec, \
        cnp.ndarray[cnp.double_t,ndim=1] dJvec1, \
        cnp.ndarray[cnp.double_t,ndim=1] dJvec2, \
        cnp.ndarray[cnp.int64_t,ndim=2] Uinds):
    """
    Sherman-Morrison block-inversion for Jitter (Cythonized)

    Calculates Trace(N^{-1} dN/dJp1 N^{-1} dN/dJp2), where:
        - N^{-1} is the ecorr-include N inverse
        - dN/dJpx = U dJ/dJpx U^{T}, with dJ/dJpx the diagnal derivative of J wrt
          Jpx

    :param Nvec:    The white noise amplitude, array (n)
    :param Jvec:    The jitter amplitude, array (k)
    :param dJvec1:  The white noise derivative, array (k)
    :param dJvec2:  The white noise derivative, array (k)
    :param Uinds:   The start/finish indices for the jitter blocks (k x 2)

    For this version, the residuals need to be sorted properly so that all the
    blocks are continuous in memory. Here, there are n residuals, and k jitter
    parameters.
    """
    cdef unsigned int cc, ii, rows = len(Nvec), cols = len(Jvec)
    cdef double tr=0.0, ji, nisum, beta
    cdef cnp.ndarray[cnp.double_t,ndim=1] ni = np.empty(rows, 'd')

    ni = 1.0 / Nvec

    for cc in range(cols):
        if Jvec[cc] > 0.0:
            ji = 1.0 / Jvec[cc]

            nisum = 0.0
            for ii in range(Uinds[cc,0],Uinds[cc,1]):
                nisum += ni[ii]

            beta = 1.0 / (nisum + ji)

            tr += dJvec1[cc] * dJvec2[cc] * nisum**2
            tr -= 2 * dJvec1[cc] * dJvec2[cc] * beta * nisum**3
            tr += dJvec1[cc] * dJvec2[cc] * beta**2 * nisum**4

    return tr

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double c_blas_block_shermor_2D_asymm(
            cnp.ndarray[cnp.double_t,ndim=2] Z1,
            cnp.ndarray[cnp.double_t,ndim=2] Z2,
            cnp.ndarray[cnp.double_t,ndim=1] Nvec,
            cnp.ndarray[cnp.double_t,ndim=1] Jvec,
            Uinds,
            cnp.ndarray[cnp.double_t,ndim=2] ZNZ,
        ):
    cdef double d_Jldet

    cdef int n_Z1_rows = Z1.shape[0]
    cdef int n_Z1_cols = Z1.shape[1]
    cdef int n_Z2_cols = Z2.shape[1]
    cdef int n_J_rows = len(Jvec)
    cdef int n_Z1_row_major, n_Z2_row_major

    n_Z1_row_major = 0 if Z1.flags['F_CONTIGUOUS'] else 1
    n_Z2_row_major = 0 if Z2.flags['F_CONTIGUOUS'] else 1

    # Hack, because somehow we can't pass integer arrays?
    # This is a tiny bit of overhead right here
    cdef int [:] Uinds_new

    # This makes it into a C-ordered (Row-Major) array we can pass
    Uinds_new = np.ascontiguousarray(Uinds.flatten(), dtype=np.dtype("i"))

    assert Z1.shape[0] == Z2.shape[0]
    assert Z1.shape[0] == len(Nvec)
    assert Uinds.shape[0] == len(Jvec)

    blas_block_shermor_2D_asym(
                 n_Z1_rows,
                 n_Z1_cols,
                 n_Z1_row_major,
                 &Z1[0,0],
                 n_Z2_cols,
                 n_Z2_row_major,
                 &Z2[0,0],
                 &Nvec[0],
                 n_J_rows,
                 &Jvec[0],
                 &Uinds_new[0],
                 &ZNZ[0,0],
                 &d_Jldet)

    return d_Jldet

def cython_blas_block_shermor_2D_asymm(Z1, Z2, Nvec, Jvec, Uinds):
    """Wrapper for the C/Cython code"""

    ZNZ = np.zeros((Z1.shape[1], Z2.shape[1]), order='F')
    Jldet = c_blas_block_shermor_2D_asymm(Z1, Z2, Nvec, Jvec, np.array(Uinds, order="C"), ZNZ)

    return Jldet, ZNZ

def cython_blas_idx_block_shermor_2D_asymm(Z1, Z2, Nvec, Jvec, Uinds, slc_isort):
    """Wrapper for the C/Cython code

    Because we have (potentially) non-contiguous blocks in memory for Z1/Z2
    we just re-order Z1 and Z2 here pre-emptively
    """
    Z1n = Z1[slc_isort,:]
    Z2n = Z2[slc_isort,:]
    Nvecn = Nvec[slc_isort]

    ZNZ = np.zeros((Z1n.shape[1], Z2n.shape[1]), order='F')
    Jldet = c_blas_block_shermor_2D_asymm(Z1n, Z2n, Nvecn, Jvec, np.array(Uinds, order="C"), ZNZ)

    return Jldet, ZNZ
