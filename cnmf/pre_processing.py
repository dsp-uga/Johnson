# -*- coding: utf-8 -*-
"""
A set of pre-processing operations in the input dataset:
1. Interpolation of missing data
2. Indentification of saturated pixels
3. Estimation of noise level for each imaged voxel
4. Estimation of global time constants

@authors: agiovann epnev
"""

import numpy as np

#%%
def interpolate_missing_data(Y):
    """
    Interpolate any missing data using nearest neighbor interpolation.
    Missing data is identified as entries with values NaN
    Input:
    Y   np.ndarray (3D)
        movie, raw data in 3D format (d1 x d2 x T)

    Outputs:
    Y   np.ndarray (3D)
        movie, data with interpolated entries (d1 x d2 x T)
    coor list
        list of interpolated coordinates
    """
    coor=[];
    if np.any(np.isnan(Y)):
        raise Exception('The algorithm has not been tested with missing values (NaNs). Remove NaNs and rerun the algorithm.')
        # need to
        for idx,row in enumerate(Y):
            nans=np.where(np.isnan(row))[0]
            n_nans=np.where(~np.isnan(row))[0]
            coor.append((idx,nans))
            Y[idx,nans]=np.interp(nans, n_nans, row[n_nans])

    return Y, coor

#%%

def get_noise_fft(Y, noise_range = [0.25,0.5], noise_method = 'logmexp', max_num_samples_fft=10000):
    """Estimate the noise level for each pixel by averaging the power spectral density.
    Inputs:
    Y: np.ndarray
    Input movie data with time in the last axis
    noise_range: np.ndarray [2 x 1] between 0 and 0.5
        Range of frequencies compared to Nyquist rate over which the power spectrum is averaged
        default: [0.25,0.5]
    noise method: string
        method of averaging the noise.
        Choices:
            'mean': Mean
            'median': Median
            'logmexp': Exponential of the mean of the logarithm of PSD (default)

    Output:
    sn: np.ndarray
        Noise level for each pixel
    """
    T = np.shape(Y)[-1]
    if T > max_num_samples_fft:
        Y=np.concatenate((Y[...,1:np.int(max_num_samples_fft/3)],        
                         Y[...,np.int(T/2-max_num_samples_fft/3/2):np.int(T/2+max_num_samples_fft/3/2)],
                         Y[...,-np.int(max_num_samples_fft/3):]),axis=-1)        

        T = np.shape(Y)[-1]
        
    dims = len(np.shape(Y))
    ff = np.arange(0,0.5+1./T,1./T)
    ind1 = ff > noise_range[0]
    ind2 = ff <= noise_range[1]
    ind = np.logical_and(ind1,ind2)    
    if dims > 1:
        xdft = np.fft.rfft(Y,axis=-1)
        psdx = (1./T)*abs(xdft)**2
        psdx[...,1:] *= 2
        sn = mean_psd(psdx[...,ind], method = noise_method)

    else:
        xdft = np.fliplr(rfft(Y))
        psdx = (1./T)*(xdft**2)
        psdx[1:] *=2
        sn = mean_psd(psdx[ind], method = noise_method)


    return sn






def mean_psd(y, method = 'logmexp'):
    """
    Averaging the PSD
    Inputs:
    y: np.ndarray
        PSD values
    method: string
        method of averaging the noise.
        Choices:
            'mean': Mean
            'median': Median
            'logmexp': Exponential of the mean of the logarithm of PSD (default)
    """

    if method == 'mean':
        mp = np.sqrt(np.mean(y/2,axis=-1))
    elif method == 'median':
        mp = np.sqrt(np.median(y/2,axis=-1))
    else:
        mp = np.log((y+1e-10)/2)
        mp = np.mean(mp,axis=-1)
        mp = np.exp(mp)
        mp = np.sqrt(mp)
#        mp = np.sqrt(np.exp(np.mean(np.log(y/2),axis=-1)))

    return mp


#%%




def preprocess_data(Y, sn = None , n_processes=4, backend='multithreading', n_pixels_per_process=100,  noise_range = [0.25,0.5], noise_method = 'logmexp', compute_g=False,  p = 2, g = None,  lags = 5, include_noise = False, pixels = None,max_num_samples_fft=10000):
    """
    Performs the pre-processing operations described above.
    """

    Y,coor=interpolate_missing_data(Y)

    if sn is None:
        sn = get_noise_fft(Y, noise_range = noise_range, noise_method = noise_method)


    return Y, sn
