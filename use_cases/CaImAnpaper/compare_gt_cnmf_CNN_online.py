#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Complete pipeline for online processing using OnACID. 

@author: Andrea Giovannucci @agiovann and Eftychios Pnevmatikakis @epnev
Special thanks to Andreas Tolias and his lab at Baylor College of Medicine 
for sharing their data used in this demo.
"""
from __future__ import division
from __future__ import print_function
import numpy as np
try:
    if __IPYTHON__:
        print('Debugging!')
        # this is used for debugging purposes only. allows to reload classes when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    print('Not IPYTHON')
    pass

from time import time
import caiman as cm
from caiman.source_extraction import cnmf as cnmf
from caiman.utils.visualization import view_patches_bar
from caiman.utils.utils import download_demo
import pylab as pl
import scipy
from caiman.motion_correction import motion_correct_iteration_fast
import cv2
from caiman.utils.visualization import plot_contours
import glob
from caiman.source_extraction.cnmf.online_cnmf import bare_initialization, initialize_movie_online, RingBuffer
#from caiman.source_extraction.cnmf.online_cnmf import load_object, save_object
from copy import deepcopy
import os

from builtins import str
from builtins import range

try:
    cv2.setNumThreads(1)
except:
    print('Open CV is naturally single threaded')


#%% Select a dataset
# 0: neuforinder.03.00.test
# 1: neurofinder.04.00.test
# 2: neurofinder.02.00
# 3: yuste
# 4: neurofinder.00.00
# 5: neurofinder,01.01
# 6: sue_ann_k53_20160530
# 7: J115
# 8: J123

ind_dataset = 8

use_VST = False
plot_results = False

#%% set some global parameters here

global_params = {'min_SNR': 2.0,        # minimum SNR when considering adding a new neuron
                 'gnb': 2,             # number of background components
                 'rval_thr': 0.80,     # spatial correlation threshold
                 # length of mini batch for OnACID in decay time units (length would be batch_length_dt*decay_time*fr)
                 'batch_length_dt': 10,
                 'max_thr': 0.30,       # parameter for thresholding components when cleaning up shapes
                 # flag for motion correction (set to False to compare directly on the same FOV)
                 'mot_corr': False,
                 'max_num_added': 1,   # maximum number of new components to be added at each timestep
                 'min_num_trial': 1    # minimum number of trials
                 }

params_movie = [{}] * 10        # set up list of dictionaries
#% neurofinder.03.00.test
params_movie[0] = {'fname': '/mnt/ceph/neuro/labeling/neurofinder.03.00.test/images/final_map/Yr_d1_498_d2_467_d3_1_order_C_frames_2250_.mmap',
                   'folder_name': '/mnt/ceph/neuro/labeling/neurofinder.03.00.test/',
                   'epochs': 2,
                   'ds_factor': 2,
                   'p': 1,  # order of the autoregressive system
                   'fr': 20,
                   'decay_time': 0.4,
                   'gSig': [11, 11],  # expected half size of neurons
                   'gnb': 2,
                   'T1': 2250
                   }

#% neurofinder.04.00.test
params_movie[1] = {'fname': '/mnt/ceph/neuro/labeling/neurofinder.04.00.test/images/final_map/Yr_d1_512_d2_512_d3_1_order_C_frames_3000_.mmap',
                   'folder_name': '/mnt/ceph/neuro/labeling/neurofinder.04.00.test/',
                   'epochs': 2,
                   'ds_factor': 1,
                   'p': 1,  # order of the autoregressive system
                   'fr': 8,
                   'gSig': [7, 7],  # expected half size of neurons
                   'decay_time': 0.5,  # rough length of a transient
                   'gnb': 2,
                   'T1': 3000,
                   }

#% neurofinder 02.00
params_movie[2] = {'fname': '/mnt/ceph/neuro/labeling/neurofinder.02.00/images/final_map/Yr_d1_512_d2_512_d3_1_order_C_frames_8000_.mmap',
                   'folder_name': '/mnt/ceph/neuro/labeling/neurofinder.02.00/',
                   'epochs': 2,
                   'ds_factor': 1,
                   'p': 1,  # order of the autoregressive system
                   'fr': 30,  # imaging rate in Hz
                   'gSig': [8, 8],  # expected half size of neuron
                   'decay_time': 0.3,
                   'gnb': 2,
                   'T1': 8000,
                   }

#% yuste
params_movie[3] = {'fname': '/mnt/ceph/neuro/labeling/yuste.Single_150u/images/final_map/Yr_d1_200_d2_256_d3_1_order_C_frames_3000_.mmap',
                   'folder_name': '/mnt/ceph/neuro/labeling/yuste.Single_150u/',
                   'epochs': 2,
                   'ds_factor': 1,
                   'p': 1,  # order of the autoregressive system
                   'fr': 10,
                   'decay_time': 0.75,
                   'T1': 3000,
                   'gnb': 2,
                   'gSig': [7, 7],  # expected half size of neurons
                   }


#% neurofinder.00.00
params_movie[4] = {'fname': '/mnt/ceph/neuro/labeling/neurofinder.00.00/images/final_map/Yr_d1_512_d2_512_d3_1_order_C_frames_2936_.mmap',
                   'folder_name':  '/mnt/ceph/neuro/labeling/neurofinder.00.00/',
                   'epochs': 2,
                   'ds_factor': 1,
                   'p': 1,  # order of the autoregressive system
                   'decay_time': 0.4,
                   'fr': 8,
                   'gSig': [8, 8],  # expected half size of neurons
                   'gnb': 2,
                   'T1': 2936,
                   }
#% neurofinder.01.01
params_movie[5] = {'fname': '/mnt/ceph/neuro/labeling/neurofinder.01.01/images/final_map/Yr_d1_512_d2_512_d3_1_order_C_frames_1825_.mmap',
                   'folder_name': '/mnt/ceph/neuro/labeling/neurofinder.01.01/',
                   'epochs': 2,
                   'ds_factor': 1,
                   'p': 1,  # order of the autoregressive system
                   'fr': 8,
                   'gnb': 1,
                   'T1': 1825,
                   'decay_time': 1.4,
                   'gSig': [6, 6]
                   }
#% Sue Ann k56
params_movie[6] = {'fname': '/mnt/ceph/neuro/labeling/k53_20160530/images/final_map/Yr_d1_512_d2_512_d3_1_order_C_frames_116043_.mmap',
                   'folder_name': '/mnt/ceph/neuro/labeling/k53_20160530/',
                   'gtname': '/mnt/ceph/neuro/labeling/k53_20160530/regions/joined_consensus_active_regions.npy',
                   'epochs': 1,
                   'ds_factor': 2,
                   'p': 1,  # order of the autoregressive system
                   'T1': 3000,  # number of frames per file
                   'fr': 30,
                   'decay_time': 0.3,
                   'gSig': [8, 8],  # expected half size of neurons
                   'gnb': 2,
                   }

#% J115
params_movie[7] = {'fname': '/mnt/ceph/neuro/labeling/J115_2015-12-09_L01_ELS/images/final_map/Yr_d1_463_d2_472_d3_1_order_C_frames_90000_.mmap',
                   'folder_name': '/mnt/ceph/neuro/labeling/J115_2015-12-09_L01_ELS/',
                   'gtname': '/mnt/ceph/neuro/labeling/J115_2015-12-09_L01_ELS/regions/joined_consensus_active_regions.npy',
                   'epochs': 1,
                   'ds_factor': 2,
                   'p': 1,  # order of the autoregressive system
                   'T1': 1000,
                   'gnb': 2,
                   'fr': 30,
                   'decay_time': 0.4,
                   'gSig': [8, 8]
                   }

#% J123
params_movie[8] = {'fname': '/mnt/ceph/neuro/labeling/J123_2015-11-20_L01_0/images/final_map/Yr_d1_458_d2_477_d3_1_order_C_frames_41000_.mmap',
                   'folder_name': '/mnt/ceph/neuro/labeling/J123_2015-11-20_L01_0/',
                   'gtname': '/mnt/ceph/neuro/labeling/J123_2015-11-20_L01_0/regions/joined_consensus_active_regions.npy',
                   'epochs': 2,
                   'ds_factor': 2,
                   'p': 1,  # order of the autoregressive system
                   'fr': 30,
                   'T1': 1000,
                   'decay_time': 0.5,
                   'gSig': [11, 11]
                   }


# % convert mmaps into tifs
#import os.path
#
# for ind_dataset in range(9):
#    fls = glob.glob(params_movie[ind_dataset]['folder_name']+'images/mmap/*.mmap')
#    for file_count, ffll in enumerate(fls):
#        file_name = '/'.join(ffll.split('/')[:-2]+['mmap_tifs']+[ffll.split('/')[-1][:-4]+'tif'])
#        if not os.path.isfile(file_name):
#            fl_temp = cm.movie(np.array(cm.load(ffll)))
#            fl_temp.save(file_name)
#        print(file_name)
#    print(ind_dataset)
#%%  download and list all files to be processed

mot_corr = global_params['mot_corr']

if mot_corr:
    working_folder = '/'.join(params_movie[ind_dataset]
                              ['fname'].split('/')[:-3] + ['images', 'tif'])
    #fls = glob.glob('/'.join( params_movie[ind_dataset]['fname'].split('/')[:-3]+['images','tifs','*.tif']))
    template = cm.load('/'.join(params_movie[ind_dataset]['fname'].split('/')[
                       :-3] + ['projections', 'median_projection.tif']))
else:
    if not use_VST:
        working_folder = '/'.join(params_movie[ind_dataset]
                                  ['fname'].split('/')[:-3] + ['images', 'mmap_tifs'])
        #fls = glob.glob('/'.join( params_movie[ind_dataset]['fname'].split('/')[:-3]+['images','mmap_tifs','*.tif']))
    else:
        working_folder = '/'.join(params_movie[ind_dataset]
                                  ['fname'].split('/')[:-3] + ['images', 'tiff_VST'])
        #fls = glob.glob('/'.join( params_movie[ind_dataset]['fname'].split('/')[:-3]+['images','tiff_VST','*.tif']))

fls = glob.glob(working_folder + '/*.tif')
fls.sort()
print(fls)

#%% Set up some parameters
# spatial downsampling factor (increases speed but may lose some fine structure)
ds_factor = params_movie[ind_dataset]['ds_factor']
# expected half size of neurons
gSig = tuple(
    np.ceil(np.array(params_movie[ind_dataset]['gSig']) / ds_factor).astype(np.int))
# number of files used for initialization
init_files = 1
# number of files used for online
online_files = len(fls) - 1
# number of frames for initialization (presumably from the first file)
initbatch = 200
# maximum number of expected components used for memory pre-allocation (exaggerate here)
expected_comps = 4000
# initial number of components
K = 2
# number of passes over the data
epochs = params_movie[ind_dataset]['epochs']
# total length of all files (if not known use a large number, then truncate at the end)
T1 = params_movie[ind_dataset]['T1'] * len(fls) * epochs

# number of timesteps to consider when testing new neuron candidates
N_samples = np.ceil(params_movie[ind_dataset]['fr']
                    * params_movie[ind_dataset]['decay_time'])
# *np.sqrt(T1/2500.)                                      # adaptive way to set threshold (will be equal to min_SNR)
min_SNR = global_params['min_SNR']
min_SNR = 2.33 * np.log(T1 / 4000 + 1.46)

# pr_inc = 1 - scipy.stats.norm.cdf(global_params['min_SNR'])           # inclusion probability of noise transient
# thresh_fitness_raw = np.log(pr_inc)*N_samples       # event exceptionality threshold
thresh_fitness_raw = scipy.special.log_ndtr(-min_SNR) * N_samples
thresh_fitness_delta = -80.                         # make this very neutral
# correlation threshold for new component inclusion
rval_thr = global_params['rval_thr']
# number of background components
gnb = global_params['gnb']
# order of AR indicator dynamics
p = params_movie[ind_dataset]['p']
deconv_flag = p > 0

#minibatch_length = int(global_params['batch_length_dt']*params_movie[ind_dataset]['fr']*params_movie[ind_dataset]['decay_time'])

#%%    Initialize movie

# load only the first initbatch frames and possibly downsample them
if ds_factor > 1:
    Y = cm.load(fls[0], subindices=slice(0, initbatch, None)).astype(
        np.float32).resize(1. / ds_factor, 1. / ds_factor)
else:
    Y = cm.load(fls[0], subindices=slice(
        0, initbatch, None)).astype(np.float32)

if mot_corr:                                        # perform motion correction on the first initbatch frames
    # maximum allowed shift during motion correction
    max_shift = np.ceil(5. / ds_factor).astype('int')
    mc = Y.motion_correct(max_shift, max_shift, template=template)
    Y = mc[0].astype(np.float32)
    borders = np.max(mc[1])
else:
    Y = Y.astype(np.float32)

# minimum value of movie. Subtract it to make the data non-negative
img_min = Y.min()
Y -= img_min
img_norm = np.std(Y, axis=0)
# normalizing factor to equalize the FOV
img_norm += np.median(img_norm)
Y = Y / img_norm[None, :, :]                        # normalize data

_, d1, d2 = Y.shape
dims = (d1, d2)                                     # dimensions of FOV
Yr = Y.to_2D().T                                    # convert data into 2D array

Cn_init = Y.local_correlations(swap_dim=False)    # compute correlation image
#pl.figure(); pl.imshow(Cn_init); pl.title('Correlation Image on initial batch'); pl.colorbar()

#%% initialize OnACID with bare initialization

cnm_init = bare_initialization(Y[:initbatch].transpose(1, 2, 0), init_batch=initbatch, k=K, gnb=gnb,
                               gSig=gSig, p=p, minibatch_shape=100, minibatch_suff_stat=5,
                               update_num_comps=True, rval_thr=rval_thr,
                               thresh_fitness_delta=thresh_fitness_delta,
                               thresh_fitness_raw=thresh_fitness_raw,
                               batch_update_suff_stat=True, max_comp_update_shape=200,
                               deconv_flag=deconv_flag,
                               simultaneously=True, n_refit=0)

cnm_init._prepare_object(np.asarray(Yr[:, :initbatch]), T1, expected_comps, idx_components=None, N_samples_exceptionality=int(N_samples),
                         max_num_added=global_params['max_num_added'], min_num_trial=global_params['min_num_trial'])

if plot_results:   # plot initialization results
    A, C, b, f, YrA, sn = cnm_init.A, cnm_init.C, cnm_init.b, cnm_init.f, cnm_init.YrA, cnm_init.sn
    view_patches_bar(Yr, scipy.sparse.coo_matrix(
        A.tocsc()[:, :]), C[:, :], b, f, dims[0], dims[1], YrA=YrA[:, :], img=Cn_init)


#%% Run OnACID and optionally plot results in real time

cnm2 = deepcopy(cnm_init)
cnm2.max_comp_update_shape = np.inf
cnm2.update_num_comps = True
t = cnm2.initbatch
tottime = []
Cn = Cn_init.copy()

# flag for plotting contours of detected components at the end of each file
plot_contours_flag = False
# flag for showing video with results online (turn off flags for improving speed)
play_reconstr = False
# flag for saving movie (file could be quite large..)
save_movie = False
movie_name = params_movie[ind_dataset]['folder_name'] + \
    'output.avi'  # name of movie to be saved
resize_fact = 1.2                        # image resizing factor

if online_files == 0:                    # check whether there are any additional files
    process_files = fls[:init_files]     # end processing at this file
    init_batc_iter = [initbatch]         # place where to start
    end_batch = T1
else:
    process_files = fls[:init_files + online_files]     # additional files
    # where to start reading at each file
    init_batc_iter = [initbatch] + [0] * online_files

shifts = []
if save_movie and play_reconstr:
    fourcc = cv2.VideoWriter_fourcc('8', 'B', 'P', 'S')
    out = cv2.VideoWriter(movie_name, fourcc, 30.0, tuple(
        [int(2 * x * resize_fact) for x in cnm2.dims]))

for iter in range(epochs):
    if iter > 0:
        # if not on first epoch process all files from scratch
        process_files = fls[:init_files + online_files]
        init_batc_iter = [0] * (online_files + init_files)      #

    # np.array(fls)[np.array([1,2,3,4,5,-5,-4,-3,-2,-1])]:
    for file_count, ffll in enumerate(process_files):
        print('Now processing file ' + ffll)
        Y_ = cm.load(ffll, subindices=slice(
            init_batc_iter[file_count], T1, None))

        # update max-correlation (and perform offline motion correction) just for illustration purposes
        if plot_contours_flag:
            if ds_factor > 1:
                Y_1 = Y_.resize(1. / ds_factor, 1. / ds_factor, 1)
            else:
                Y_1 = Y_.copy()
                if mot_corr:
                    templ = (cnm2.Ab.data[:cnm2.Ab.indptr[1]] * cnm2.C_on[0,
                                                                          t - 1]).reshape(cnm2.dims, order='F') * img_norm
                    newcn = (Y_1 - img_min).motion_correct(max_shift, max_shift,
                                                           template=templ)[0].local_correlations(swap_dim=False)
                    Cn = np.maximum(Cn, newcn)
                else:
                    Cn = np.maximum(Cn, Y_1.local_correlations(swap_dim=False))

        old_comps = cnm2.N                              # number of existing components
        for frame_count, frame in enumerate(Y_):        # now process each file
            if np.isnan(np.sum(frame)):
                raise Exception('Frame ' + str(frame_count) + ' contains nan')
            if t % 100 == 0:
                print('Epoch: ' + str(iter + 1) + '. ' + str(t) + ' frames have beeen processed in total. ' + str(cnm2.N -
                                                                                                                  old_comps) + ' new components were added. Total number of components is ' + str(cnm2.Ab.shape[-1] - gnb))
                old_comps = cnm2.N

            t1 = time()                                 # count time only for the processing part
            frame_ = frame.copy().astype(np.float32)    #
            if ds_factor > 1:
                # downsample if necessary
                frame_ = cv2.resize(frame_, img_norm.shape[::-1])

            frame_ -= img_min                                       # make data non-negative

            if mot_corr:                                            # motion correct
                templ = cnm2.Ab.dot(
                    cnm2.C_on[:cnm2.M, t - 1]).reshape(cnm2.dims, order='F') * img_norm
                frame_cor, shift = motion_correct_iteration_fast(
                    frame_, templ, max_shift, max_shift)
                shifts.append(shift)
            else:
                templ = None
                frame_cor = frame_

            frame_cor = frame_cor / img_norm                        # normalize data-frame
            cnm2.fit_next(t, frame_cor.reshape(-1, order='F')
                          )      # run OnACID on this frame
            # store time
            tottime.append(time() - t1)

            t += 1
            # if t>=4500:
            #    break

            if t % 1000 == 0 and plot_contours_flag:
                pl.cla()
                A = cnm2.Ab[:, cnm2.gnb:]
                # update the contour plot every 1000 frames
                crd = cm.utils.visualization.plot_contours(A, Cn, thr=0.9)
                pl.pause(1)

            if play_reconstr:                                               # generate movie with the results
                A, b = cnm2.Ab[:, cnm2.gnb:], cnm2.Ab[:, :cnm2.gnb].toarray()
                C, f = cnm2.C_on[cnm2.gnb:cnm2.M, :], cnm2.C_on[:cnm2.gnb, :]
                # inferred activity due to components (no background)
                comps_frame = A.dot(
                    C[:, t - 1]).reshape(cnm2.dims, order='F') * img_norm / np.max(img_norm)
                # denoised frame (components + background)
                bgkrnd_frame = b.dot(
                    f[:, t - 1]).reshape(cnm2.dims, order='F') * img_norm / np.max(img_norm)
                # spatial shapes
                all_comps = (np.array(A.sum(-1)).reshape(cnm2.dims, order='F'))
                frame_comp_1 = cv2.resize(np.concatenate([frame_ / np.max(img_norm), all_comps * 3.], axis=-1), (2 * np.int(
                    cnm2.dims[1] * resize_fact), np.int(cnm2.dims[0] * resize_fact)))
                frame_comp_2 = cv2.resize(np.concatenate([comps_frame * 10., comps_frame + bgkrnd_frame], axis=-1), (2 * np.int(
                    cnm2.dims[1] * resize_fact), np.int(cnm2.dims[0] * resize_fact)))
                frame_pn = np.concatenate(
                    [frame_comp_1, frame_comp_2], axis=0).T
                vid_frame = np.repeat(frame_pn[:, :, None], 3, axis=-1)
                vid_frame = np.minimum((vid_frame * 255.), 255).astype('u1')
                cv2.putText(vid_frame, 'Raw Data', (5, 20), fontFace=5,
                            fontScale=1.2, color=(0, 255, 0), thickness=1)
                cv2.putText(vid_frame, 'Inferred Activity', (np.int(
                    cnm2.dims[0] * resize_fact) + 5, 20), fontFace=5, fontScale=1.2, color=(0, 255, 0), thickness=1)
                cv2.putText(vid_frame, 'Identified Components', (5, np.int(
                    cnm2.dims[1] * resize_fact) + 20), fontFace=5, fontScale=1.2, color=(0, 255, 0), thickness=1)
                cv2.putText(vid_frame, 'Denoised Data', (np.int(cnm2.dims[0] * resize_fact) + 5, np.int(
                    cnm2.dims[1] * resize_fact) + 20), fontFace=5, fontScale=1.2, color=(0, 255, 0), thickness=1)
                cv2.putText(vid_frame, 'Frame = ' + str(t), (vid_frame.shape[1] // 2 - vid_frame.shape[1] //
                                                             10, vid_frame.shape[0] - 20), fontFace=5, fontScale=1.2, color=(0, 255, 255), thickness=1)
                if save_movie:
                    out.write(vid_frame)
                cv2.imshow('frame', vid_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        print('Cumulative processing speed is ' + str((t - initbatch) /
                                                      np.sum(tottime))[:5] + ' frames per second.')

if save_movie:
    out.release()
cv2.destroyAllWindows()

#%% extract results from the objects and do some plotting
A, b = cnm2.Ab[:, cnm2.gnb:], cnm2.Ab[:, :cnm2.gnb].toarray()
C, f = cnm2.C_on[cnm2.gnb:cnm2.M, t - t //
                 epochs:t], cnm2.C_on[:cnm2.gnb, t - t // epochs:t]
noisyC = cnm2.noisyC[:, t - t // epochs:t]

if deconv_flag:
    b_trace = [osi.b for osi in cnm2.OASISinstances]


if plot_results:
    pl.figure()
    crd = cm.utils.visualization.plot_contours(A, Cn, thr=0.9)
    view_patches_bar(Yr, scipy.sparse.coo_matrix(A.tocsc()[:, :]), C[:, :], b, f,
                     dims[0], dims[1], YrA=noisyC[cnm2.gnb:cnm2.M] - C, img=Cn)


#%%  save results (optional)

save_results = False
save_file = working_folder + '/results_analysis_online_IND_' + str(ind_dataset) + '_VST_' + str(use_VST) + '_rval_' + str(
    int(10 * global_params['rval_thr'])) + '_minSNR_' + str(int(10 * global_params['min_SNR'])) + '.npz'
if save_results:
    np.savez(save_file,
             Cn=Cn, A=A, b=b, Cf=cnm2.C_on, f=cnm2.f,
             dims=cnm2.dims, tottime=tottime, noisyC=cnm2.noisyC, shifts=shifts, img=Cn,
             params_movie=params_movie, global_params=global_params, T1=T1, t=t)

#%%  load_results (start from here if loading an already processed files)

load_results = True

if load_results:
    load_file = save_file
    if os.path.exists(load_file):
        with np.load(load_file) as ld:
            print(ld.keys())
            locals().update(ld)
            global_params = global_params[()]
            params_movie = params_movie[()]
            A = A[()]

#%% load, threshold and filter for size ground truth
#global_params['max_thr'] = 0.2
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=None, single_thread=True)

gt_file = os.path.join(os.path.split(params_movie[ind_dataset]['fname'])[
                       0], os.path.split(params_movie[ind_dataset]['fname'])[1][:-4] + 'match_masks.npz')
min_radius = min(gSig[0] / 2., 3.)          # minimum acceptable radius
max_radius = 2. * gSig[0]                  # maximum acceptable radius
min_size_neuro = min_radius**2 * np.pi
max_size_neuro = max_radius**2 * np.pi

with np.load(gt_file, encoding='latin1') as ld:
    print(ld.keys())
    d1_or = int(ld['d1'])
    d2_or = int(ld['d2'])
    dims_or = (d1_or, d2_or)
    A_gt = ld['A_gt'][()].toarray()
    Cn_orig = ld['Cn']
    # locals().update(ld)
    #A_gt = scipy.sparse.coo_matrix(A_gt[()])
    #dims = (d1,d2)

if ds_factor > 1:
    A_gt = cm.movie(np.reshape(A_gt, dims_or + (-1,), order='F')
                    ).transpose(2, 0, 1).resize(1. / ds_factor, 1. / ds_factor)
    pl.figure()
    pl.imshow(A_gt.sum(0))
    A_gt = np.array(np.reshape(A_gt, (A_gt.shape[0], -1), order='F')).T
    Cn_orig = cv2.resize(Cn_orig, None, fx=1. / ds_factor, fy=1. / ds_factor)

A_gt_thr = cm.source_extraction.cnmf.spatial.threshold_components(A_gt, dims, medw=None, thr_method='max', nrgthr=0.95, maxthr=global_params['max_thr'], extract_cc=True,
                                                                  se=None, ss=None, dview=None)

A_gt_thr_bin = A_gt_thr > 0
size_neurons_gt = A_gt_thr_bin.sum(0)
idx_size_neurons_gt = np.where(
    (size_neurons_gt > min_size_neuro) & (size_neurons_gt < max_size_neuro))[0]
print(A_gt_thr.shape)
#%% filter for size found neurons

A_thr = cm.source_extraction.cnmf.spatial.threshold_components(A.tocsc()[:, :].toarray(), dims, medw=None, thr_method='max', nrgthr=0.95, maxthr=global_params['max_thr'], extract_cc=True,
                                                               se=None, ss=None, dview=dview)

A_thr_bin = A_thr > 0
size_neurons = A_thr_bin.sum(0)
idx_size_neurons = np.where(
    (size_neurons > min_size_neuro) & (size_neurons < max_size_neuro))[0]
#A_thr = A_thr[:,idx_size_neuro]
print(A_thr.shape)

#%% compute results

use_cnn = False  # Use CNN classifier
if use_cnn:
    from caiman.components_evaluation import evaluate_components_CNN
    predictions, final_crops = evaluate_components_CNN(
        A, dims, gSig, model_name='use_cases/CaImAnpaper/cnn_model')
    thresh_cnn = .05
    idx_components_cnn = np.where(predictions[:, 1] >= thresh_cnn)[0]
    idx_neurons = np.intersect1d(idx_components_cnn, idx_size_neurons)
else:
    idx_neurons = idx_size_neurons.copy()

if plot_results:
    pl.figure(figsize=(30, 20))

tp_gt, tp_comp, fn_gt, fp_comp, performance_cons_off = cm.base.rois.nf_match_neurons_in_binary_masks(A_gt_thr_bin[:, idx_size_neurons_gt].reshape([dims[0], dims[1], -1], order='F').transpose([2, 0, 1]) * 1.,
                                                                                                     A_thr_bin[:, idx_neurons].reshape([dims[0], dims[1], -1], order='F').transpose([2, 0, 1]) * 1., thresh_cost=.7, min_dist=10,
                                                                                                     print_assignment=False, plot_results=plot_results, Cn=Cn_orig, labels=['GT', 'Offline'], enclosed_thr=None)

pl.rcParams['pdf.fonttype'] = 42
font = {'family': 'Arial',
        'weight': 'regular',
        'size': 20}

pl.rc('font', **font)
print({a: b.astype(np.float16) for a, b in performance_cons_off.items()})
