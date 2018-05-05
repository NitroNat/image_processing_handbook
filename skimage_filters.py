from skimage import (io,
                     data,
                     color,
                     util,
                    filters,
                     )
import os, glob
from matplotlib import pyplot as plt

image_ironchef = io.imread('./images/ironchef.jpg')
image_grey = color.rgb2gray(image_ironchef)
image_grey = util.img_as_ubyte(image_grey)

# Apply Filter
fig, axes = plt.subplots(1,2)
fig.suptitle('Mr. Chairman from Iron Chef')
axes[0].imshow(image_grey, cmap=plt.cm.gray)
axes[0].axis('off')
axes[0].set_title('GrayScale')
axes[1].imshow( filters.sobel(image_grey) , cmap=plt.cm.gray)
axes[1].axis('off')
axes[1].set_title('Edges')
plt.show()