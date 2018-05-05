from skimage import (io,
                     data,
                     color,
                     util,
                    filters,
                    feature,
                     )
import os, glob
from matplotlib import pyplot as plt
import numpy as np

coins = data.coins()

# Thresholding
fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
axes[0].imshow(coins > 100, cmap=plt.cm.gray, interpolation='nearest')
axes[0].set_title('coins > 100')
axes[1].imshow(coins > 150, cmap=plt.cm.gray, interpolation='nearest')
axes[1].set_title('coins > 150')
for a in axes:
    a.axis('off')

# Edge-Based Segmentation
edges = feature.canny(coins)
from scipy import ndimage as ndi
fill_coins = ndi.binary_fill_holes(edges)

# label all the objects
label_objects, nb_labels = ndi.label(fill_coins)
# count the number of pixels for each object
sizes = np.bincount(label_objects.ravel())
mask_sizes = sizes > 20 # Keep all the big area objects
mask_sizes[0] = 0
coins_cleaned = mask_sizes[label_objects]

fig, axes = plt.subplots(1,2)
fig.suptitle('Coins Segmentation')
axes[0].imshow(coins, cmap=plt.cm.gray)
axes[0].axis('off')
axes[0].set_title('GrayScale')
axes[1].hist( coins.flatten() , bins=np.arange(0,256))
axes[1].axis('on')


fig, axes = plt.subplots(1,2)
fig.suptitle('Coins Segmentation')
axes[0].imshow(edges, cmap=plt.cm.gray)
axes[0].axis('on')
axes[0].set_title('Edges')
axes[1].imshow(fill_coins, cmap=plt.cm.gray)
axes[1].axis('on')
axes[1].set_title('Fill Holes')


fig, axes = plt.subplots(1,1)
fig.suptitle('Coins Segmentation')
axes.imshow(coins_cleaned, cmap=plt.cm.gray)
axes.axis('on')
axes.set_title('Objects')

# Show all the plots at once
plt.show()