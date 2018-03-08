# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 15:53:15 2016

@author: agiovann
"""
from __future__ import division
from __future__ import print_function
from builtins import zip
from builtins import str
from builtins import map
from builtins import range
from past.utils import old_div
import cv2
try:
    cv2.setNumThreads(1)
except:
    print('Open CV is naturally single threaded')

try:
    if __IPYTHON__:
        print((1))
        # this is used for debugging purposes only. allows to reload classes when changed
        get_ipython().magic('load_ext autoreload')
        get_ipython().magic('autoreload 2')
except NameError:
    print('Not IPYTHON')
    pass
#%%
import caiman as cm
import numpy as np
import os
import glob
import time
import pylab as pl
import psutil
import sys
from ipyparallel import Client
from skimage.external.tifffile import TiffFile
import scipy
import glob
#%%
from caiman.motion_correction import tile_and_correct, motion_correction_piecewise
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.components_evaluation import evaluate_components
from caiman.utils.visualization import plot_contours, view_patches_bar
from caiman.base.rois import extract_binary_masks_blob


#%%
params_movie = {'fname': None,
                'max_shifts': (6, 6),  # maximum allow rigid shift
                'splits_rig': 56,  # for parallelization split the movies in  num_splits chuncks across time
                # if none all the splits are processed and the movie is saved
                'num_splits_to_process_rig': None,
                # intervals at which patches are laid out for motion correction
                'strides': (128, 128),
                # overlap between pathes (size of patch strides+overlaps)
                'overlaps': (32, 32),
                'splits_els': 56,  # for parallelization split the movies in  num_splits chuncks across time
                # if none all the splits are processed and the movie is saved
                'num_splits_to_process_els': [28, None],
                'upsample_factor_grid': 4,  # upsample factor to avoid smearing when merging patches
                # maximum deviation allowed for patch with respect to rigid shift
                'max_deviation_rigid': 3,
                'p': 1,  # order of the autoregressive system
                'merge_thresh': 0.8,  # merging threshold, max correlation allowed
                # half-size of the patches in pixels. rf=25, patches are 50x50
                'rf': (10, 30),
                # amount of overlap between the patches in pixels
                'stride_cnmf': (5, 10),
                'K': 6,  # number of components per patch
                # if dendritic. In this case you need to set init_method to sparse_nmf
                'is_dendrites': True,
                'init_method': 'sparse_nmf',
                'gSig': [0, 0],  # expected half size of neurons
                'alpha_snmf': 10,  # this controls sparsity
                'final_frate': 30
                }
#%%
#m_orig = cm.load(params_movie['fname'])
#%% start local cluster
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=None, single_thread=False)
#%%
if params_movie['fname'] is None:
    all_files = [os.path.abspath(flfl) for flfl in glob.glob('*.tif')]
    all_files.sort()

else:
    all_files = [params_movie['fname']]

print(all_files)
#%% RIGID MOTION CORRECTION
total_template_rig = None
total_shifts = []
templates_all = []
add_to_movie = np.min(cm.load(all_files[0]))
max_shifts = params_movie['max_shifts']  # maximum allowed shifts
num_iter = 1  # number of times the algorithm is run
# for parallelization split the movies in  num_splits chuncks across time
splits = params_movie['splits_rig']
# if none all the splits are processed and the movie is saved
num_splits_to_process = params_movie['num_splits_to_process_rig']
shifts_opencv = True  # apply shifts fast way (but smoothing results)
save_movie_rigid = False  # save the movies vs just get the template
for file_to_process in all_files:
    t1 = time.time()
    fname = file_to_process

    t1 = time.time()
    fname_tot_rig, total_template_rig, templates_rig, shifts_rig = cm.motion_correction.motion_correct_batch_rigid(fname,
                                                                                                                   max_shifts, dview=dview, splits=splits, num_splits_to_process=num_splits_to_process,
                                                                                                                   num_iter=num_iter,  template=total_template_rig, shifts_opencv=shifts_opencv, save_movie_rigid=save_movie_rigid, add_to_movie=add_to_movie)
    total_shifts.append(shifts_rig)
    templates_all.append(total_template_rig)
    t2 = time.time() - t1
    print(t2)
    pl.imshow(total_template_rig, cmap='gray',
              vmax=np.percentile(total_template_rig, 95))
    pl.pause(1)


#%%
pl.close()
pl.plot(np.concatenate(total_shifts, 0))
#%% visualize all templates
cm.movie(np.array(templates_all)).play(fr=2, gain=5)

#%% PIECEWISE RIGID MOTION CORRECTION
total_shifts_els = []
templates_all_els = []
new_templ = total_template_rig.copy()
strides = params_movie['strides']
overlaps = params_movie['overlaps']
shifts_opencv = True
save_movie = True
splits = params_movie['splits_els']
num_splits_to_process_list = params_movie['num_splits_to_process_els']
upsample_factor_grid = params_movie['upsample_factor_grid']
max_deviation_rigid = params_movie['upsample_factor_grid']

for file_to_process in all_files:
    t1 = time.time()

    num_iter = 1
    for num_splits_to_process in num_splits_to_process_list:
        fname_tot_els, total_template_wls, templates_els, x_shifts_els, y_shifts_els, coord_shifts_els = cm.motion_correction.motion_correct_batch_pwrigid(file_to_process, max_shifts, strides, overlaps, add_to_movie, newoverlaps=None,  newstrides=None,
                                                                                                                                                           dview=dview, upsample_factor_grid=upsample_factor_grid, max_deviation_rigid=max_deviation_rigid,
                                                                                                                                                           splits=splits, num_splits_to_process=num_splits_to_process, num_iter=num_iter,
                                                                                                                                                           template=new_templ, shifts_opencv=shifts_opencv, save_movie=save_movie)
        new_templ = total_template_wls

    if len(num_splits_to_process_list) > 1:
        num_splits_to_process_list = [None]

    total_shifts_els.append([x_shifts_els, y_shifts_els])
    templates_all_els.append(total_template_wls)

#%%
pl.subplot(2, 1, 1)
pl.plot(np.concatenate([shfts[0] for shfts in total_shifts_els], 0))
pl.subplot(2, 1, 2)
pl.plot(np.concatenate([shfts[1] for shfts in total_shifts_els], 0))
#%%
border_to_0 = np.ceil(np.max(np.array(total_shifts_els))).astype(np.int)
#%%
fnames_map = [os.path.abspath(flfl) for flfl in glob.glob('*.mmap')]
fnames_map.sort()
print(fnames_map)

#%%
# add_to_movie=np.nanmin(templates_rig)+1# the movie must be positive!!!
downsample_factor = 1  # use .2 or .1 if file is large and you want a quick answer
idx_xy = None
base_name = 'Yr'
name_new = cm.save_memmap_each(fnames_map, dview=dview, base_name=base_name, resize_fact=(
    1, 1, downsample_factor), remove_init=0, idx_xy=idx_xy, add_to_movie=add_to_movie, border_to_0=border_to_0)
name_new.sort()
print(name_new)
#%%
if len(name_new) > 1:
    fname_new = cm.save_memmap_join(
        name_new, base_name='Yr', n_chunks=12, dview=dview)
else:
    print('One file only, not saving!')
    fname_new = name_new[0]
#%%
Yr, dims, T = cm.load_memmap(fname_new)
d1, d2 = dims
images = np.reshape(Yr.T, [T] + list(dims), order='F')
Y = np.reshape(Yr, dims + (T,), order='F')
#%%
Cn = cm.local_correlations(Y[:, :, :3000])
pl.imshow(Cn, cmap='gray')

#%%
if np.min(images) < 0:
    raise Exception('Movie too negative, add_to_movie should be larger')
if np.sum(np.isnan(images)) > 0:
    raise Exception('Movie contains nan! You did not remove enough borders')
#%%
m_els = cm.load(fname_tot_els)
downs = .2
m_els.resize(1, 1, downs).play(
    fr=30, gain=5, magnification=1, offset=add_to_movie)
#%% some parameter settings
p = params_movie['p']  # order of the autoregressive system
# merging threshold, max correlation allowed
merge_thresh = params_movie['merge_thresh']
# half-size of the patches in pixels. rf=25, patches are 50x50
rf = params_movie['rf']
# amounpl.it of overlap between the patches in pixels
stride = params_movie['stride_cnmf']
K = params_movie['K']  # number of neurons expected per patch
gSig = params_movie['gSig']
init_method = params_movie['init_method']
alpha_snmf = params_movie['alpha_snmf']

if params_movie['is_dendrites'] == True:
    if params_movie['init_method'] is not 'sparse_nmf':
        raise Exception('dendritic requires sparse_nmf')
    if params_movie['alpha_snmf'] is None:
        raise Exception('need to set a value for alpha_snmf')
#%%
t1 = time.time()
cnm = cnmf.CNMF(n_processes, k=K, gSig=gSig, merge_thresh=0.8, p=0, dview=dview, Ain=None, rf=rf, stride=stride, memory_fact=1,
                method_init=init_method, alpha_snmf=alpha_snmf, only_init_patch=True, gnb=1, method_deconvolution='oasis')
cnm = cnm.fit(images)

A_tot = cnm.A
C_tot = cnm.C
YrA_tot = cnm.YrA
b_tot = cnm.b
f_tot = cnm.f
sn_tot = cnm.sn
t2 = time.time() - t1
print(('Number of components:' + str(A_tot.shape[-1])))
#%%
pl.figure()
crd = plot_contours(A_tot, Cn, thr=0.9)
#%% DISCARD LOW QUALITY COMPONENT
t1 = time.time()
final_frate = params_movie['final_frate']
r_values_min = .6  # threshold on space consistency
fitness_min = -30  # threshold on time variability
# threshold on time variability (if nonsparse activity)
fitness_delta_min = -30
Npeaks = 10
traces = C_tot + YrA_tot
idx_components, idx_components_bad = cm.components_evaluation.estimate_components_quality(
    traces, Y, A_tot, C_tot, b_tot, f_tot, final_frate=final_frate, Npeaks=Npeaks, r_values_min=r_values_min, fitness_min=fitness_min, fitness_delta_min=fitness_delta_min)
t2 = time.time() - t1
print(('Keeping ' + str(len(idx_components)) +
       ' and discarding  ' + str(len(idx_components_bad))))
#%%
pl.figure()
crd = plot_contours(A_tot.tocsc()[:, idx_components], Cn, thr=0.9)
#%%
A_tot = A_tot.tocsc()[:, idx_components]
C_tot = C_tot[idx_components]
#%% rerun updating the components
cnm = cnmf.CNMF(n_processes, k=A_tot.shape, gSig=gSig, merge_thresh=merge_thresh, p=p, dview=dview, Ain=A_tot, Cin=C_tot,
                f_in=f_tot, rf=None, stride=None, method_deconvolution='oasis')
cnm = cnm.fit(images)
#%%
A, C, b, f, YrA, sn = cnm.A, cnm.C, cnm.b, cnm.f, cnm.YrA, cnm.sn
#%% again recheck quality of components, stricter criteria
final_frate = params_movie['final_frate']
r_values_min = .75
fitness_min = - 50
fitness_delta_min = - 50
Npeaks = 10
traces = C + YrA
idx_components, idx_components_bad = cm.components_evaluation.estimate_components_quality(
    traces, Y, A, C, b, f, final_frate=final_frate, Npeaks=Npeaks, r_values_min=r_values_min, fitness_min=fitness_min, fitness_delta_min=fitness_delta_min)
print(' ***** ')
print((len(traces)))
print((len(idx_components)))
#%% save results
np.savez(os.path.join(os.path.split(fname_new)[0], os.path.split(fname_new)[1][:-4] + 'results_analysis_3.npz'), Cn=Cn, A=A.todense(
), C=C, b=b, f=f, YrA=YrA, sn=sn, d1=d1, d2=d2, idx_components=idx_components, idx_components_bad=idx_components_bad)
#%%
pl.subplot(1, 2, 1)
crd = plot_contours(A.tocsc()[:, idx_components], Cn, thr=0.9)
pl.subplot(1, 2, 2)
crd = plot_contours(A.tocsc()[:, idx_components_bad], Cn, thr=0.9)
#%%
view_patches_bar(Yr, scipy.sparse.coo_matrix(A.tocsc()[:, idx_components]), C[
    idx_components, :], b, f, dims[0], dims[1], YrA=YrA[idx_components, :], img=Cn)
#%%
view_patches_bar(Yr, scipy.sparse.coo_matrix(A.tocsc()[:, idx_components_bad]), C[
    idx_components_bad, :], b, f, dims[0], dims[1], YrA=YrA[idx_components_bad, :], img=Cn)
#%% STOP CLUSTER and clean up log files
cm.stop_server()

import glob
log_files = glob.glob('Yr*_LOG_*')
for log_file in log_files:
    os.remove(log_file)
