import thunder as td
from extraction import NMF

def main(images, _k=5, _percentile=99, _max_iter=50, _overlap=0.1, _chunk_size = 32, _padding = 25, _merge = 0.1):
    '''
    Main method for NMF approach. This is a wrapper built upon the original pipeline of NMF in Thunder Extraction.
    Images are images after preprocessing.
    '''
    algorithm = NMF(k=_k, percentile=_percentile, max_iter=_max_iter, overlap=_overlap)
    model = algorithm.fit(images, chunk_size=(_chunk_size,_chunk_size), padding=(_padding,_padding))
    merged = model.merge(_merge)
    regions = merged.regions
    return regions
