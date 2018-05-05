'''
Rank filters are non-linear filters using the local gray-level ordering to compute the filtered value. This ensemble
of filters share a common base: the local gray-level histogram is computed on the neighborhood of a pixel (defined by a 2-D structuring element). If the filtered value is taken as the middle value of the histogram, we get the classical median filter.

Rank filters can be used for several purposes such as:

image quality enhancement e.g. image smoothing, sharpening
image pre-processing e.g. noise reduction, contrast enhancement
feature extraction e.g. border detection, isolated point detection
post-processing e.g. small object removal, object grouping, contour smoothing
Some well known filters are specific cases of rank filters [1] e.g. morphological dilation, morphological erosion, median filters.

Rank filters can be faster than morpholical operations
The central part of the skimage.rank filters is build on a sliding window that updates the local gray-level histogram. This approach limits the algorithm complexity to O(n) where n is the number of image pixels. The complexity is also limited with respect to the structuring element size.
'''

import numpy as np
import matplotlib.pyplot as plt

from skimage import img_as_ubyte
from skimage import data

noisy_image = img_as_ubyte(data.camera())
hist = np.histogram(noisy_image, bins=np.arange(0, 256))

fig, ax = plt.subplots(ncols=2, figsize=(10, 5))

ax[0].imshow(noisy_image, interpolation='nearest', cmap=plt.cm.gray)
ax[0].axis('off')

ax[1].plot(hist[1][:-1], hist[0], lw=2)
ax[1].set_title('Histogram of grey values')

plt.tight_layout()

# Removing Salt 'N Pepper Noise
# Smoothing
from skimage.filters.rank import median # Median Filter
from skimage.morphology import disk

# Add some noise
noise = np.random.random(noisy_image.shape)
noisy_image = img_as_ubyte(data.camera())
noisy_image[noise > 0.99] = 255
noisy_image[noise < 0.01] = 0

fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(noisy_image, vmin=0, vmax=255, cmap=plt.cm.gray)
ax[0].set_title('Noisy image')

ax[1].imshow(median(noisy_image, disk(1)), vmin=0, vmax=255, cmap=plt.cm.gray)
ax[1].set_title('Median $r=1$')

ax[2].imshow(median(noisy_image, disk(5)), vmin=0, vmax=255, cmap=plt.cm.gray)
ax[2].set_title('Median $r=5$')

ax[3].imshow(median(noisy_image, disk(20)), vmin=0, vmax=255, cmap=plt.cm.gray)
ax[3].set_title('Median $r=20$')

for a in ax:
    a.axis('off')

plt.tight_layout()

# Smoothing Color Images
from skimage.filters.rank import mean_bilateral

noisy_image = img_as_ubyte(data.camera())
bilat = mean_bilateral(noisy_image.astype(np.uint16), disk(20), s0=10, s1=10)

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10),
                         sharex='row', sharey='row')
ax = axes.ravel()
ax[0].imshow(noisy_image, cmap=plt.cm.gray)
ax[0].set_title('Original')
ax[1].imshow(bilat, cmap=plt.cm.gray)
ax[1].set_title('Bilateral mean')
ax[2].imshow(noisy_image[200:350, 350:450], cmap=plt.cm.gray)
ax[3].imshow(bilat[200:350, 350:450], cmap=plt.cm.gray)
for a in ax:
    a.axis('off')
plt.tight_layout()

# Contrast Enhancement
from skimage import exposure
from skimage.filters import rank

noisy_image = img_as_ubyte(data.camera())

# equalize globally and locally
glob = exposure.equalize_hist(noisy_image) * 255
loc = rank.equalize(noisy_image, disk(20))

# extract histogram for each image
hist = np.histogram(noisy_image, bins=np.arange(0, 256))
glob_hist = np.histogram(glob, bins=np.arange(0, 256))
loc_hist = np.histogram(loc, bins=np.arange(0, 256))

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 12))
ax = axes.ravel()
ax[0].imshow(noisy_image, interpolation='nearest', cmap=plt.cm.gray)
ax[0].axis('off')
ax[1].plot(hist[1][:-1], hist[0], lw=2)
ax[1].set_title('Histogram of gray values')
ax[2].imshow(glob, interpolation='nearest', cmap=plt.cm.gray)
ax[2].axis('off')
ax[3].plot(glob_hist[1][:-1], glob_hist[0], lw=2)
ax[3].set_title('Histogram of gray values')
ax[4].imshow(loc, interpolation='nearest', cmap=plt.cm.gray)
ax[4].axis('off')
ax[5].plot(loc_hist[1][:-1], loc_hist[0], lw=2)
ax[5].set_title('Histogram of gray values')

plt.tight_layout()

# Local-based contrast enhancement
from skimage.filters.rank import autolevel

noisy_image = img_as_ubyte(data.camera())
auto = autolevel(noisy_image.astype(np.uint16), disk(20))
fig, ax = plt.subplots(ncols=2, figsize=(10, 5), sharex=True, sharey=True)

ax[0].imshow(noisy_image, cmap=plt.cm.gray)
ax[0].set_title('Original')
ax[1].imshow(auto, cmap=plt.cm.gray)
ax[1].set_title('Local autolevel')
for a in ax:
    a.axis('off')

plt.tight_layout()

# Morphological Contrast Enhancements
from skimage.filters.rank import enhance_contrast

noisy_image = img_as_ubyte(data.camera())
enh = enhance_contrast(noisy_image, disk(5))
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10),
                         sharex='row', sharey='row')
ax = axes.ravel()
ax[0].imshow(noisy_image, cmap=plt.cm.gray)
ax[0].set_title('Original')
ax[1].imshow(enh, cmap=plt.cm.gray)
ax[1].set_title('Local morphological contrast enhancement')
ax[2].imshow(noisy_image[200:350, 350:450], cmap=plt.cm.gray)
ax[3].imshow(enh[200:350, 350:450], cmap=plt.cm.gray)

for a in ax:
    a.axis('off')

plt.tight_layout()

# Morphological Contrast Enhancements with percentiles
from skimage.filters.rank import enhance_contrast_percentile

noisy_image = img_as_ubyte(data.camera())

penh = enhance_contrast_percentile(noisy_image, disk(5), p0=.1, p1=.9)

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10),
                         sharex='row', sharey='row')
ax = axes.ravel()
ax[0].imshow(noisy_image, cmap=plt.cm.gray)
ax[0].set_title('Original')
ax[1].imshow(penh, cmap=plt.cm.gray)
ax[1].set_title('Local percentile morphological\n contrast enhancement')
ax[2].imshow(noisy_image[200:350, 350:450], cmap=plt.cm.gray)
ax[3].imshow(penh[200:350, 350:450], cmap=plt.cm.gray)

for a in ax:
    a.axis('off')

plt.tight_layout()


# ##############################################################################
'''
Feature extraction
Local histograms can be exploited to compute local entropy, which is related to the local image complexity. Entropy is computed using base 2 logarithm i.e. the filter returns the minimum number of bits needed to encode local gray-level distribution.

skimage.rank.entropy() returns the local entropy on a given structuring element. The following example shows applies this filter on 8- and 16-bit images.
'''
from skimage import data
from skimage.filters.rank import entropy
from skimage.morphology import disk
import numpy as np
import matplotlib.pyplot as plt

image = data.camera()

fig, ax = plt.subplots(ncols=2, figsize=(12, 6), sharex=True, sharey=True)

fig.colorbar(ax[0].imshow(image, cmap=plt.cm.gray), ax=ax[0])
ax[0].set_title('Image')

fig.colorbar(ax[1].imshow(entropy(image, disk(5)), cmap=plt.cm.gray), ax=ax[1])
ax[1].set_title('Entropy')

for a in ax:
    a.axis('off')

plt.tight_layout()