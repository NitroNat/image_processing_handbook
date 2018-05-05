from skimage import (io,
                     data,
                     color,
                     util,
                     )
import os, glob
from matplotlib import pyplot as plt

# Load images
image_ironchef = io.imread('./images/ironchef.jpg')
image_camera = data.camera()

# Write Images
io.imsave('./ironchef.jpg', image_ironchef, quality=100) # save 2D array as image file
io.imsave('./camera.jpg', image_camera, quality=100) # save 2D array as image file

files = [f for f in os.listdir('.') if os.path.isfile(f)]
for f in files:
    print(f)

# Load multiple images in a collections object
file_names = glob.glob('./*.jpg')
images = io.imread_collection(file_names)
image1 = images[0]
image2 = images[1]

print(image1.shape, type(image1), image1.dtype)
print(image2.shape, type(image2), image2.dtype)

image2_grey = color.rgb2gray(image2)
image2_grey = util.img_as_ubyte(image2_grey)
print(image2_grey.shape, type(image2_grey), image2_grey.dtype)
print(image2_grey[:10])

# Display images
fig, axes = plt.subplots(1,3)
fig.suptitle('Mr. Chairman from Iron Chef')
axes[0].imshow(image2)
axes[0].axis('off')
axes[0].set_title('Colour')
axes[1].imshow(image2_grey, cmap=plt.cm.gray)
axes[1].axis('off')
axes[1].set_title('GrayScale')
axes[2].imshow(util.invert(image2_grey), cmap=plt.cm.gray)
axes[2].axis('off')
axes[2].set_title('GrayScale - Inverted')
plt.tight_layout()
plt.show()