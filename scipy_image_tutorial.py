# Tutorial on image processing using scipy

from scipy import misc
import matplotlib.pyplot as plt 
import numpy as np
import imageio # good package for reading and writing image data

f = misc.face() # get a racoon face
imageio.imwrite('face.png', f) # save 2D array as image file
racoon = imageio.imread('face.png')
#plt.imshow(racoon); plt.show()
# Data is a numpy array
#print(racoon.shape) # depth for R,G,B
#print(racoon.dtype) # 8 bit images 0-255

# Opening raw image files
racoon.tofile('face.raw') # Create raw image
racoon_from_raw = np.fromfile('face.raw',dtype=np.uint8)
print(racoon_from_raw.shape)
# Need to reshape
racoon_from_raw.reshape(768, 1024, 3)
# use memmap for large arrays

# Displaying image
#plt.imshow(racoon, cmap=plt.cm.gray); 
#plt.axis('off')
#plt.show()








