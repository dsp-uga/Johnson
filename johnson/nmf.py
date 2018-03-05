import thunder as td
from extraction import NMF
import johnson

def main(setName=['00.00', '00.01','01.00','01.01','02.00','02.01','03.00','04.00','04.01'],
            base='caesar', _k=5, _percentile=99, _max_iter=50, _overlap=0.1, _chunk_size=32,
            _padding=25, _merge=0.1):
    '''
    Main method for NMF approach. This is a wrapper built upon the original pipeline of NMF in Thunder Extraction.
    The code for putting data into json files is from:
    https://gist.github.com/freeman-lab/330183fdb0ea7f4103deddc9fae18113
    '''
    submission = []
    for data in setName:
        images = johnson.load(setName, base)
        images = johnson.grayScale(images)
        print ('The shape of each training image after preprocessing is {}'.format(images[1].shape))
        print ('Applying median filter for {}.test'.format(data))
        images = johnson.medianFilter(images)
        print ('Applying NMF for {}.test.....'.format(data))
        algorithm = NMF(k=_k, percentile=_percentile, max_iter=_max_iter, overlap=_overlap)
        model = algorithm.fit(images, chunk_size=(_chunk_size,_chunk_size), padding=(_padding,_padding))
        print ('Merge regions for {}.test....'.format(data))
        merged = model.merge(_merge)
        regions = [{'coordinates': region.coordinates.tolist()} for region in merged.regions]
        result = {'dataset': '{}.test'.format(data), 'regions': regions}
        submission.append(result)

        # show a message for processing
        print ('Completed processing results for {}.test'.format(data))

    with open('{}.json'.format('submission'), 'w') as f:
        f.write(json.dumps(submission))
    print ('Done!')
