import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import io
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.filters import roberts, sobel, scharr, prewitt
from skimage.feature import canny
from skimage import restoration

dftrain = pd.read_csv('train.csv')
imgSize = 256
petIds = dftrain['PetID'].values

img = io.imread('../train_img/'+petIds[0]+'-1.jpg', as_gray=True)

rescaled_img = rescale(img, 1.0/4.0, anti_aliasing=True)
resized_img = resize(img, (200,200))
downscaled_img = downscale_local_mean(img, (4,3))

edge_roberts = roberts(img)
edge_sobel = sobel(img)
edge_scharr = scharr(img)
edge_prewitt = prewitt(img)

edge_canny = canny(img, sigma=3)

psf = np.ones((3,3)) / 9
deconvolved, _ = restoration.unsupervised_wiener(img, psf)

print(edge_prewitt)

plt.imshow(edge_prewitt)
plt.show()