
## fit a model
import thunder as td
from regional import one, many
import numpy as np
import matplotlib.pyplot as plot
from showit import image
from fakearray import calcium_imaging

from cnmf import CNMF



data, series, truth = calcium_imaging(n=5, t=10, seed=42, noise=0.5, withparams=True)
base = data.mean(0)
image(base, size=10);
plot.show()

algorithm = CNMF( k=5, gSig=[4,4], merge_thresh=0.8)

model,temporaldata = algorithm.fit(data)

def convert(array):
    r,c = np.where(array > 0.0)
    return one(zip(r,c))

regions = many([convert(model[:,:,i]) for i in range(model.shape[2])])

#show true solution
image(many(truth).mask(dims=data.shape[1:], cmap='rainbow', stroke='black', base=base));
plot.show()

#show algorithm solution
masks = regions.mask(cmap_stroke='rainbow', fill=None, base=base.clip(0,4000) / 4000)
image(masks, size=14);
plot.show()
