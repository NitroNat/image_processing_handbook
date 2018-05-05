'''
Morphological image processing is a collection of non-linear operations related to the shape or morphology of features in an image, such as boundaries, skeletons, etc. In any given technique, we probe an image with a small shape or template called a structuring element, which defines the region of interest or neighborhood around a pixel.
'''

import os
import matplotlib.pyplot as plt
from skimage.data import data_dir
from skimage.util import img_as_ubyte
from skimage import io

orig_phantom = img_as_ubyte(io.imread(os.path.join(data_dir, "phantom.png"),
                                      as_grey=True))
fig, ax = plt.subplots()
ax.imshow(orig_phantom, cmap=plt.cm.gray)

def plot_comparison(original, filtered, filter_name):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True,
                                   sharey=True)
    ax1.imshow(original, cmap=plt.cm.gray)
    ax1.set_title('original')
    ax1.axis('off')
    ax2.imshow(filtered, cmap=plt.cm.gray)
    ax2.set_title(filter_name)
    ax2.axis('off')

# Erosion - shrink foreground
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import black_tophat, skeletonize, convex_hull_image
from skimage.morphology import disk

eroded = erosion(orig_phantom, disk(6)) # structuring element
plot_comparison(orig_phantom, eroded, 'erosion')

# Dilation - enlarge foreground
dilated = dilation(orig_phantom, disk(6))
plot_comparison(orig_phantom, dilated, 'dilation')

# Opening - erosion, then dilation (remove "salt")
opened = opening(orig_phantom, disk(6))
plot_comparison(orig_phantom, opened, 'opening')

# Closing - dilation, then erosion (remove "pepper")
phantom = orig_phantom.copy()
phantom[10:30, 200:210] = 0

closed = closing(phantom, disk(6))
plot_comparison(phantom, closed, 'closing')

# White Tophat
'''
The white_tophat of an image is defined as the image minus its morphological opening. This operation returns the bright spots of the image that are smaller than the structuring element.
'''
phantom = orig_phantom.copy()
# Add a black and white spot
phantom[340:350, 200:210] = 255
phantom[100:110, 200:210] = 0

w_tophat = white_tophat(phantom, disk(6))
plot_comparison(phantom, w_tophat, 'white tophat')

# Black Tophat
'''
The black_tophat of an image is defined as its morphological closing minus the original image. 
This operation returns the dark spots of the image that are smaller than the structuring element.
'''
b_tophat = black_tophat(phantom, disk(6))
plot_comparison(phantom, b_tophat, 'black tophat')

# Convex Hull
'''
The convex_hull_image is the set of pixels included in the smallest convex 
polygon that surround all white pixels in the input image. 
Again note that this is also performed on binary images.
'''
horse = io.imread(os.path.join(data_dir, "horse.png"), as_grey=True)
hull1 = convex_hull_image(horse == 0)
plot_comparison(horse, hull1, 'convex hull')