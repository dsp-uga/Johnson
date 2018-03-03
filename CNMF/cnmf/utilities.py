# -*- coding: utf-8 -*-
"""A set of utilities, mostly for post-processing and visualization
Created on Sat Sep 12 15:52:53 2015

@author epnev
"""

import numpy as np
from scipy.sparse import spdiags
#import ca_source_extraction

#%%


def CNMFSetParms(Y, n_processes, K=30, gSig=[5, 5], ssub=1, tsub=1, p=2, p_ssub=1, p_tsub=1, thr=0.8,backend='single_thread', **kwargs):
    """Dictionary for setting the CNMF parameters.
    Any parameter that is not set get a default value specified
    by the dictionary default options
    """

    if type(Y) is tuple:
        dims, T = Y[:-1], Y[-1]
    else:
        dims, T = Y.shape[:-1], Y.shape[-1]

    print ('using ' + str(n_processes) + ' processes')
    n_pixels_per_process = np.prod(dims) / n_processes  # how to subdivide the work among processes

    options = dict()
    options['patch_params'] = {
        'ssub': p_ssub,             # spatial downsampling factor
        'tsub': p_tsub              # temporal downsampling factor
    }
    options['preprocess_params'] = {'sn': None,                  # noise level for each pixel
                                    # range of normalized frequencies over which to average
                                    'noise_range': [0.25, 0.5],
                                    # averaging method ('mean','median','logmexp')
                                    'noise_method': 'logmexp',
                                    'max_num_samples_fft': 3000,
                                    'n_processes': n_processes,
                                    'n_pixels_per_process': n_pixels_per_process,
                                    'compute_g': False,            # flag for estimating global time constant
                                    'p': p,                        # order of AR indicator dynamics
                                    'lags': 5,                     # number of autocovariance lags to be considered for time constant estimation
                                    'include_noise': False,        # flag for using noise values when estimating g
                                    'pixels': None,                 # pixels to be excluded due to saturation
                                    'backend': backend
                                    }
    options['init_params'] = {'K': K,                                          # number of components
                              # size of components (std of Gaussian)
                              'gSig': gSig,
                              # size of bounding box
                              'gSiz': list(np.array(gSig, dtype=int) * 2 + 1),
                              'ssub': ssub,             # spatial downsampling factor
                              'tsub': tsub,             # temporal downsampling factor
                              'nIter': 5,               # number of refinement iterations
                              'kernel': None,           # user specified template for greedyROI
                              'maxIter': 5              # number of HALS iterations
                              }
    options['spatial_params'] = {
        'dims': dims,                   # number of rows, columns [and depths]
        # method for determining footprint of spatial components ('ellipse' or 'dilate')
        'method': 'ellipse',
        'dist': 3,                       # expansion factor of ellipse
        'n_processes': n_processes,      # number of process
        'n_pixels_per_process': n_pixels_per_process,    # number of pixels to be processed by eacg worker
        'backend': backend,
    }
    options['temporal_params'] = {
        'ITER': 2,                   # block coordinate descent iterations
        # method for solving the constrained deconvolution problem ('cvx' or 'cvxpy')
        'method': 'cvxpy',
        # if method cvxpy, primary and secondary (if problem unfeasible for approx
        # solution) solvers to be used with cvxpy, can be 'ECOS','SCS' or 'CVXOPT'
        'solvers': ['ECOS', 'SCS'],
        'p': p,                      # order of AR indicator dynamics
        'n_processes': n_processes,
        'backend': backend,
        'memory_efficient': False,
        # flag for setting non-negative baseline (otherwise b >= min(y))
        'bas_nonneg': True,
        # range of normalized frequencies over which to average
        'noise_range': [.25, .5],
        'noise_method': 'logmexp',   # averaging method ('mean','median','logmexp')
                        'lags': 5,                   # number of autocovariance lags to be considered for time constant estimation
                        # bias correction factor (between 0 and 1, close to 1)
                        'fudge_factor': .98,
                        'verbosity': False
    }
    options['merging'] = {
        'thr': thr,
    }
    return options

#%%

def update_order(A):
    '''Determines the update order of the temporal components given the spatial
    components by creating a nest of random approximate vertex covers
     Input:
     -------
     A:    np.ndarray
          matrix of spatial components (d x K)

     Outputs:
     ---------
     O:   list of sets
          list of subsets of components. The components of each subset can be updated in parallel
     lo:  list
          length of each subset

    Written by Eftychios A. Pnevmatikakis, Simons Foundation, 2015
    '''
    K = np.shape(A)[-1]
    AA = A.T * A
    AA.setdiag(0)
    F = (AA) > 0
    F = F.toarray()
    rem_ind = np.arange(K)
    O = []
    lo = []
    while len(rem_ind) > 0:
        L = np.sort(app_vertex_cover(F[rem_ind, :][:, rem_ind]))
        if L.size:
            ord_ind = set(rem_ind) - set(rem_ind[L])
            rem_ind = rem_ind[L]
        else:
            ord_ind = set(rem_ind)
            rem_ind = []

        O.append(ord_ind)
        lo.append(len(ord_ind))

    return O[::-1], lo[::-1]

def local_correlations(Y, eight_neighbours=True, swap_dim=True):
    """Computes the correlation image for the input dataset Y

    Parameters
    -----------

    Y:  np.ndarray (3D or 4D)
        Input movie data in 3D or 4D format
    eight_neighbours: Boolean
        Use 8 neighbors if true, and 4 if false for 3D data (default = True)
        Use 6 neighbors for 4D data, irrespectively
    swap_dim: Boolean
        True indicates that time is listed in the last axis of Y (matlab format)
        and moves it in the front

    Returns
    --------

    rho: d1 x d2 [x d3] matrix, cross-correlation with adjacent pixels

    """

    if swap_dim:
        Y = np.transpose(Y, tuple(np.hstack((Y.ndim - 1, range(Y.ndim)[:-1]))))

    rho = np.zeros(np.shape(Y)[1:])
    w_mov = (Y - np.mean(Y, axis=0)) / np.std(Y, axis=0)

    rho_h = np.mean(np.multiply(w_mov[:, :-1, :], w_mov[:, 1:, :]), axis=0)
    rho_w = np.mean(np.multiply(w_mov[:, :, :-1], w_mov[:, :, 1:]), axis=0)

    rho[:-1, :] = rho[:-1, :] + rho_h
    rho[1:, :] = rho[1:, :] + rho_h
    rho[:, :-1] = rho[:, :-1] + rho_w
    rho[:, 1:] = rho[:, 1:] + rho_w

    if Y.ndim == 4:
        rho_d = np.mean(np.multiply(w_mov[:, :, :, :-1], w_mov[:, :, :, 1:]), axis=0)
        rho[:, :, :-1] = rho[:, :, :-1] + rho_d
        rho[:, :, 1:] = rho[:, :, 1:] + rho_d
        neighbors = 6 * np.ones(np.shape(Y)[1:])
        neighbors[0] = neighbors[0] - 1
        neighbors[-1] = neighbors[-1] - 1
        neighbors[:, 0] = neighbors[:, 0] - 1
        neighbors[:, -1] = neighbors[:, -1] - 1
        neighbors[:, :, 0] = neighbors[:, :, 0] - 1
        neighbors[:, :, -1] = neighbors[:, :, -1] - 1

    else:
        if eight_neighbours:
            rho_d1 = np.mean(np.multiply(w_mov[:, 1:, :-1], w_mov[:, :-1, 1:, ]), axis=0)
            rho_d2 = np.mean(np.multiply(w_mov[:, :-1, :-1], w_mov[:, 1:, 1:, ]), axis=0)
            rho[:-1, :-1] = rho[:-1, :-1] + rho_d2
            rho[1:, 1:] = rho[1:, 1:] + rho_d1
            rho[1:, :-1] = rho[1:, :-1] + rho_d1
            rho[:-1, 1:] = rho[:-1, 1:] + rho_d2

            neighbors = 8 * np.ones(np.shape(Y)[1:3])
            neighbors[0, :] = neighbors[0, :] - 3
            neighbors[-1, :] = neighbors[-1, :] - 3
            neighbors[:, 0] = neighbors[:, 0] - 3
            neighbors[:, -1] = neighbors[:, -1] - 3
            neighbors[0, 0] = neighbors[0, 0] + 1
            neighbors[-1, -1] = neighbors[-1, -1] + 1
            neighbors[-1, 0] = neighbors[-1, 0] + 1
            neighbors[0, -1] = neighbors[0, -1] + 1
        else:
            neighbors = 4 * np.ones(np.shape(Y)[1:3])
            neighbors[0, :] = neighbors[0, :] - 1
            neighbors[-1, :] = neighbors[-1, :] - 1
            neighbors[:, 0] = neighbors[:, 0] - 1
            neighbors[:, -1] = neighbors[:, -1] - 1

    rho = np.divide(rho, neighbors)

    return rho


def order_components(A, C):
    """Order components based on their maximum temporal value and size

    Parameters
    -----------
    A:   sparse matrix (d x K)
         spatial components
    C:   matrix or np.ndarray (K x T)
         temporal components

    Returns
    -------
    A_or:  np.ndarray
        ordered spatial components
    C_or:  np.ndarray
        ordered temporal components
    srt:   np.ndarray
        sorting mapping

    """
    A = np.array(A.todense())
    nA2 = np.sqrt(np.sum(A**2, axis=0))
    K = len(nA2)
    A = np.array(np.matrix(A) * spdiags(1 / nA2, 0, K, K))
    nA4 = np.sum(A**4, axis=0)**0.25
    C = np.array(spdiags(nA2, 0, K, K) * np.matrix(C))
    mC = np.ndarray.max(np.array(C), axis=1)
    srt = np.argsort(nA4 * mC)[::-1]
    A_or = A[:, srt] * spdiags(nA2[srt], 0, K, K)
    C_or = spdiags(1. / nA2[srt], 0, K, K) * (C[srt, :])

    return A_or, C_or, srt


def app_vertex_cover(A):
    ''' Finds an approximate vertex cover for a symmetric graph with adjacency matrix A.

     Parameters
     -----------
     A:    boolean 2d array (K x K)
          Adjacency matrix. A is boolean with diagonal set to 0

     Returns
     --------
     L:   A vertex cover of A
     Written by Eftychios A. Pnevmatikakis, Simons Foundation, 2015
    '''

    L = []
    while A.any():
        nz = np.nonzero(A)[0]          # find non-zero edges
        u = nz[np.random.randint(0, len(nz))]
        A[u, :] = False
        A[:, u] = False
        L.append(u)

    return np.asarray(L)

