import numpy as np
from cnmf.utilities import local_correlations, CNMFSetParms, order_components
from cnmf.pre_processing import preprocess_data
from cnmf.initialization import initialize_components
from cnmf.merging import merge_components
from cnmf.spatial import update_spatial_components
from cnmf.temporal import update_temporal_components

class CNMF(object):
  """
  Source extraction using constrained non-negative matrix factorization.
  """
  def __init__(self, k=5, gSig=[4,4], merge_thresh=0.8 , p=2, backend='single_thread', n_processes=1):
      self.k = k #number of neurons expected per patch
      self.gSig=gSig # expected half size of neurons
      self.merge_thresh=merge_thresh # merging threshold, max correlation allowed
      self.p=p #order of the autoregressive system
      self.backend=backend
      self.n_processes=n_processes
      

  def fit(self, images):
  	"""
  	This method uses the cnmf algorithm to find sources in data.

  	Parameters
  	----------
  	images : np.ndarray
		Array of shape (t,x,y) containing the images that vary over time.


	Returns
    --------
    neurons : np.ndarray
    	Array of shape (x,y,n) where n is the neuron number
	temporaldata : np.ndarray
		Array of shape (n,t) where n is the neuron number and t is the frame number
  	"""
  	dims=(images.shape[1],images.shape[2])
  	T=images.shape[0]
  	Yr = np.transpose(images, range(1, len(dims) + 1) + [0])
  	Yr = np.reshape(Yr, (np.prod(dims), T), order='F')
  	Y=np.reshape(Yr,dims+(T,),order='F')
  	options = CNMFSetParms(Y,p=self.p,gSig=self.gSig,K=self.k, backend=self.backend, thr=self.merge_thresh, n_processes=self.n_processes)
  	
  	Cn = local_correlations(Y)
  	Yr,sn = preprocess_data(Yr,**options['preprocess_params'])
  	Atmp, Ctmp, b_in, f_in, center=initialize_components(Y, **options['init_params'])
  	Ain,Cin = Atmp, Ctmp
  	A,b,Cin = update_spatial_components(Yr, Cin, f_in, Ain, sn=sn, **options['spatial_params'])  
  	options['temporal_params']['p'] = 0 # set this to zero for fast updating without deconvolution
  	C,f,S,bl,c1,neurons_sn,g,YrA = update_temporal_components(Yr,A,b,Cin,f_in,bl=None,c1=None,sn=None,g=None,**options['temporal_params'])
  	A_m,C_m,nr_m,merged_ROIs,S_m,bl_m,c1_m,sn_m,g_m=merge_components(Yr,A,b,C,f,S,sn,options['temporal_params'],options['spatial_params'], options['merging'],bl=bl, c1=c1, sn=neurons_sn, g=g, mx=50, fast_merge = True)
  	A2,b2,C2 = update_spatial_components(Yr, C_m, f, A_m, sn=sn, **options['spatial_params'])
  	options['temporal_params']['p'] = self.p # set it back to original value to perform full deconvolution
  	C2,f2,S2,bl2,c12,neurons_sn2,g21,YrA = update_temporal_components(Yr,A2,b2,C2,f,bl=None,c1=None,sn=None,g=None,**options['temporal_params'])
  	A_or, temporaldata, srt = order_components(A2,C2)
  	neurons=A_or.reshape(dims[1],dims[0],A_or.shape[1])
  	return np.transpose(neurons, [1,0,2]), temporaldata


  #     images = check_images(images)
  #     chunk_size = chunk_size if chunk_size is not None else images.shape[1:]
  #     blocks = images.toblocks(chunk_size=chunk_size, padding=padding)
  #     sources = asarray(blocks.map_generic(self._get))

  #     # add offsets based on block coordinates
  #     for inds in itertools.product(*[range(d) for d in sources.shape]):
  #         offset = (asarray(inds) * asarray(blocks.blockshape)[1:])
  #         for source in sources[inds]:
  #             source.coordinates += offset
  #             if padding:
  #               leftpad = [blocks.padding[i + 1] if inds[i] != 0 else 0 for i in range(len(inds))]
  #               source.coordinates -= asarray(leftpad)
      
  #     # flatten list and create model
  #     flattened = list(itertools.chain.from_iterable(sources.flatten().tolist()))
  #     return ExtractionModel(many(flattened))

  # def _get(self, block):
  #     """
  #     Perform NMF on a block to identify spatial regions.
  #     """
  #     dims = block.shape[1:]
  #     max_size = prod(dims) / 2 if self.max_size == 'full' else self.max_size

  #     # reshape to t x spatial dimensions
  #     data = block.reshape(block.shape[0], -1)

  #     # build and apply NMF model to block
  #     model = SKNMF(self.k, max_iter=self.max_iter)
  #     model.fit(clip(data, 0, inf))

  #     # reconstruct sources as spatial objects in one array
  #     components = model.components_.reshape((self.k,) + dims)

  #     # convert from basis functions into shape
  #     # by median filtering (optional), applying a percentile threshold,
  #     # finding connected components and removing small objects
  #     combined = []
  #     for component in components:
  #         tmp = component > percentile(component, self.percentile)
  #         labels, num = label(tmp, return_num=True)
  #         if num == 1:
  #           counts = bincount(labels.ravel())
  #           if counts[1] < self.min_size:
  #             continue
  #           else:
  #             regions = labels
  #         else:
  #           regions = remove_small_objects(labels, min_size=self.min_size)
  #         ids = unique(regions)
  #         ids = ids[ids > 0]
  #         for ii in ids:
  #             r = regions == ii
  #             r = median_filter(r, 2)
  #             coords = asarray(where(r)).T
  #             if (size(coords) > 0) and (size(coords) < max_size):
  #                 combined.append(one(coords))

  #     # merge overlapping sources
  #     if self.overlap is not None:

  #         # iterate over source pairs and find a pair to merge
  #         def merge(sources):
  #             for i1, s1 in enumerate(sources):
  #                 for i2, s2 in enumerate(sources[i1+1:]):
  #                     if s1.overlap(s2) > self.overlap:
  #                         return i1, i1 + 1 + i2
  #             return None

  #         # merge pairs until none left to merge
  #         pair = merge(combined)
  #         testing = True
  #         while testing:
  #             if pair is None:
  #                 testing = False
  #             else:
  #                 combined[pair[0]] = combined[pair[0]].merge(combined[pair[1]])
  #                 del combined[pair[1]]
  #                 pair = merge(combined)

  #     return combined