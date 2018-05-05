from skimage import (io,
                     data,
                     color,
                     util,
                     )
import os, glob
from matplotlib import pyplot as plt

image_ironchef = io.imread('./images/ironchef.jpg')

# Change color spaces
fig, axes = plt.subplots(1,2)
fig.suptitle('Mr. Chairman from Iron Chef')
axes[0].imshow(image_ironchef)
axes[0].axis('off')
axes[0].set_title('RGB')
axes[1].imshow(color.rgb2hsv(image_ironchef))
axes[1].axis('off')
axes[1].set_title('HSV')
plt.show()