# Tutorial on image processing using scipy

from scipy import misc
import matplotlib.pyplot as plt 
import numpy as np
import imageio # good package for reading and writing image data

f = misc.face() # get a racoon face
imageio.imwrite('face.png', f) # save 2D array as image file
racoon = imageio.imread('face.png')
plt.imshow(racoon); plt.axis('off'); plt.show()
# Data is a numpy array
print(racoon.shape) # depth for R,G,B
print(racoon.dtype) # 8 bit images 0-255

# Opening raw image files
racoon.tofile('face.raw') # Create raw image
racoon_from_raw = np.fromfile('face.raw',dtype=np.uint8)
print(racoon_from_raw.shape)
# Need to reshape
racoon_from_raw = racoon_from_raw.reshape(768, 1024, 3)
# use memmap for large arrays
face_memmap = np.memmap('./face.raw', dtype=np.uint8, shape=(768, 1024, 3))

# Displaying image
f = misc.face(gray=True) # get a racoon face
plt.imshow(f, cmap=plt.cm.gray, vmin=30, vmax=200)
plt.contour(f)
plt.axis('off')
plt.show()

# Working on a list of image files
'''
for i in range(10):
    im = np.random.randint(0, 256, 10000).reshape((100, 100))
    misc.imsave('random_%02d.png' % i, im)
from glob import glob
filelist = glob('random*.png')
filelist.sort()
'''

# Interpolation
plt.imshow(f[320:340, 510:530], cmap=plt.cm.gray, interpolation='bilinear')
plt.imshow(f[320:340, 510:530], cmap=plt.cm.gray, interpolation='nearest')

# Indexing
face = misc.face(gray=True)
face[0, 40]

# Slicing
face[10:13, 20:23]
face[100:120] = 255

lx, ly = face.shape
X, Y = np.ogrid[0:lx, 0:ly]
mask = (X - lx / 2) ** 2 + (Y - ly / 2) ** 2 > lx * ly / 4
# Masks
face[mask] = 0
# Fancy indexing
face[range(400), range(400)] = 255

# Geometric Transformations
face = misc.face(gray=True)
lx, ly = face.shape
# Cropping
crop_face = face[lx // 4: - lx // 4, ly // 4: - ly // 4]
plt.imshow(crop_face, cmap=plt.cm.gray)
# up <-> down flip
flip_ud_face = np.flipud(face)
plt.imshow(flip_ud_face, cmap=plt.cm.gray)
# rotation
from scipy import ndimage
rotate_face = ndimage.rotate(face, 45)
rotate_face_noreshape = ndimage.rotate(face, 45, reshape=False)
plt.imshow(rotate_face_noreshape, cmap=plt.cm.gray)

# Image Filtering
from scipy import misc
face = misc.face(gray=True)
blurred_face = ndimage.gaussian_filter(face, sigma=3)
plt.imshow(blurred_face, cmap=plt.cm.gray)

very_blurred = ndimage.gaussian_filter(face, sigma=5)
plt.imshow(very_blurred, cmap=plt.cm.gray)

local_mean = ndimage.uniform_filter(face, size=11)
plt.imshow(local_mean, cmap=plt.cm.gray)

from scipy import misc
face = misc.face(gray=True).astype(float)
blurred_f = ndimage.gaussian_filter(face, 3)
filter_blurred_f = ndimage.gaussian_filter(blurred_f, 1)
alpha = 30
sharpened = blurred_f + alpha * (blurred_f - filter_blurred_f)
plt.imshow(sharpened, cmap=plt.cm.gray)

# Denoise
from scipy import misc
f = misc.face(gray=True)
f = f[230:290, 220:320]
noisy = f + 0.4 * f.std() * np.random.random(f.shape)

gauss_denoised = ndimage.gaussian_filter(noisy, 2)

med_denoised = ndimage.median_filter(noisy, 3)


# Morpology
selem = ndimage.generate_binary_structure(2, 1)
selem.astype(np.int)
a = np.zeros((7,7), dtype=np.int)
a[1:6, 2:5] = 1
print(a)
print(ndimage.binary_erosion(a).astype(a.dtype))
print(ndimage.binary_erosion(a, structure=np.ones((5,5))).astype(a.dtype))

# Feature Extraction
im = np.zeros((256, 256))
im[64:-64, 64:-64] = 1
im = ndimage.rotate(im, 15, mode='constant')
im = ndimage.gaussian_filter(im, 8)
plt.imshow(im, cmap=plt.cm.gray)
sx = ndimage.sobel(im, axis=0, mode='constant')
sy = ndimage.sobel(im, axis=1, mode='constant')
sob = np.hypot(sx, sy)
plt.imshow(sob)

# Segmentation
# Histogram-based
n = 10
l = 256
im = np.zeros((l, l))
np.random.seed(1)
points = l*np.random.random((2, n**2))
im[(points[0]).astype(np.int), (points[1]).astype(np.int)] = 1
im = ndimage.gaussian_filter(im, sigma=l/(4.*n))

mask = (im > im.mean()).astype(np.float)
mask += 0.1 * im
img = mask + 0.2*np.random.randn(*mask.shape)

hist, bin_edges = np.histogram(img, bins=60)
bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])

binary_img = img > 0.5
plt.imshow(binary_img, cmap=plt.cm.gray)

# Region Properties
n = 10
l = 256
im = np.zeros((l, l))
points = l*np.random.random((2, n**2))
im[(points[0]).astype(np.int), (points[1]).astype(np.int)] = 1
im = ndimage.gaussian_filter(im, sigma=l/(4.*n))
mask = im > im.mean()
plt.imshow(im, cmap=plt.cm.gray)
# Label foreground objects
label_im, nb_labels = ndimage.label(mask)
print(nb_labels) # how many objects?
plt.imshow(label_im)

sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))
mean_vals = ndimage.sum(im, label_im, range(1, nb_labels + 1))
# Keep big objects
mask_size = sizes < 1000
remove_pixel = mask_size[label_im]
remove_pixel.shape

label_im[remove_pixel] = 0
plt.imshow(label_im)

labels = np.unique(label_im)
label_im = np.searchsorted(labels, label_im)

# Locate one of the objects
slice_x, slice_y = ndimage.find_objects(label_im==4)[0]
roi = im[slice_x, slice_y]
plt.imshow(roi)