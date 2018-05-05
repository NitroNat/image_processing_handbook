'''
Skeletonization reduces binary objects to 1 pixel wide representations.
This can be useful for feature extraction, and/or representing an object’s topology

works by making successive passes of the image, removing pixels on object borders. This continues until no more pixels can be removed. The image is correlated with a mask that assigns each pixel a number in the range [0…255] corresponding to each possible pattern of its 8 neighbouring pixels. A look up table is then used to assign the pixels a value of 0, 1, 2 or 3, which are selectively removed during the iterations.
'''
from skimage.morphology import skeletonize
from skimage import data
import matplotlib.pyplot as plt
from skimage.util import invert

# Invert the horse image
image = invert(data.horse())

# perform skeletonization
skeleton = skeletonize(image)

# display results
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4),
                         sharex=True, sharey=True)
ax = axes.ravel()
ax[0].imshow(image, cmap=plt.cm.gray)
ax[0].axis('off')
ax[0].set_title('original', fontsize=20)
ax[1].imshow(skeleton, cmap=plt.cm.gray)
ax[1].axis('off')
ax[1].set_title('skeleton', fontsize=20)
fig.tight_layout()
plt.show()

# ##############################################################################
# 3D Skele
# ##############################################################################
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize, skeletonize_3d
from skimage.data import binary_blobs

data = binary_blobs(200, blob_size_fraction=.2, volume_fraction=.35, seed=1)

skeleton = skeletonize(data)
skeleton3d = skeletonize_3d(data) # Use this for 3d images

fig, axes = plt.subplots(1, 3, figsize=(8, 4), sharex=True, sharey=True)
ax = axes.ravel()
ax[0].imshow(data, cmap=plt.cm.gray, interpolation='nearest')
ax[0].set_title('original')
ax[0].axis('off')
ax[1].imshow(skeleton, cmap=plt.cm.gray, interpolation='nearest')
ax[1].set_title('skeletonize')
ax[1].axis('off')
ax[2].imshow(skeleton3d, cmap=plt.cm.gray, interpolation='nearest')
ax[2].set_title('skeletonize_3d')
ax[2].axis('off')
fig.tight_layout()
plt.show()
# ##############################################################################
# Medial axis skeletonization
# ##############################################################################
from skimage.morphology import medial_axis, skeletonize, skeletonize_3d

# Generate the data
data = binary_blobs(200, blob_size_fraction=.2, volume_fraction=.35, seed=1)

# Compute the medial axis (skeleton) and the distance transform
skel, distance = medial_axis(data, return_distance=True)

# Compare with other skeletonization algorithms
skeleton = skeletonize(data)
skeleton3d = skeletonize_3d(data)

# Distance to the background for pixels of the skeleton
dist_on_skel = distance * skel

from skimage.util.colormap import magma

fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
ax = axes.ravel()
ax[0].imshow(data, cmap=plt.cm.gray, interpolation='nearest')
ax[0].set_title('original')
ax[0].axis('off')
ax[1].imshow(dist_on_skel, cmap=magma, interpolation='nearest')
ax[1].contour(data, [0.5], colors='w')
ax[1].set_title('medial_axis')
ax[1].axis('off')
ax[2].imshow(skeleton, cmap=plt.cm.gray, interpolation='nearest')
ax[2].set_title('skeletonize')
ax[2].axis('off')
ax[3].imshow(skeleton3d, cmap=plt.cm.gray, interpolation='nearest')
ax[3].set_title('skeletonize_3d')
ax[3].axis('off')

fig.tight_layout()
plt.show()

# ##############################################################################
# Morphological Thinning similar to skeletonization
# ##############################################################################
from skimage.morphology import skeletonize, thin

skeleton = skeletonize(image)
thinned = thin(image)
thinned_partial = thin(image, max_iter=25)

fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
ax = axes.ravel()
ax[0].imshow(image, cmap=plt.cm.gray, interpolation='nearest')
ax[0].set_title('original')
ax[0].axis('off')
ax[1].imshow(skeleton, cmap=plt.cm.gray, interpolation='nearest')
ax[1].set_title('skeleton')
ax[1].axis('off')
ax[2].imshow(thinned, cmap=plt.cm.gray, interpolation='nearest')
ax[2].set_title('thinned')
ax[2].axis('off')
ax[3].imshow(thinned_partial, cmap=plt.cm.gray, interpolation='nearest')
ax[3].set_title('partially thinned')
ax[3].axis('off')
fig.tight_layout()
plt.show()