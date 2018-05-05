'''
In computed tomography, the tomography reconstruction problem is to obtain a tomographic slice image from a set of projections. A projection is formed by drawing a set of parallel rays through the 2D object of interest, assigning the integral of the object’s contrast along each ray to a single pixel in the projection. A single projection of a 2D object is one dimensional. To enable computed tomography reconstruction of the object, several projections must be acquired, each of them corresponding to a different angle between the rays with respect to the object. A collection of projections at several angles is called a sinogram, which is a linear transform of the original image.

The inverse Radon transform is used in computed tomography to reconstruct a 2D image from the measured projections (the sinogram). A practical, exact implementation of the inverse Radon transform does not exist, but there are several good approximate algorithms available.

As the inverse Radon transform reconstructs the object from a set of projections, the (forward) Radon transform can be used to simulate a tomography experiment.

This script performs the Radon transform to simulate a tomography experiment and reconstructs the input image based on the resulting sinogram formed by the simulation. Two methods for performing the inverse Radon transform and reconstructing the original image are compared: The Filtered Back Projection (FBP) and the Simultaneous Algebraic Reconstruction Technique (SART).

As our original image, we will use the Shepp-Logan phantom. When calculating the Radon transform, we need to decide how many projection angles we wish to use. As a rule of thumb, the number of projections should be about the same as the number of pixels there are across the object (to see why this is so, consider how many unknown pixel values must be determined in the reconstruction process and compare this to the number of measurements provided by the projections), and we follow that rule here.
'''

import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage import data_dir
from skimage.transform import radon, rescale

image = imread(data_dir + "/phantom.png", as_grey=True)
image = rescale(image, scale=0.4, mode='reflect')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))

ax1.set_title("Original")
ax1.imshow(image, cmap=plt.cm.Greys_r)

theta = np.linspace(0., 180., max(image.shape), endpoint=False)
sinogram = radon(image, theta=theta, circle=True)
ax2.set_title("Radon transform\n(Sinogram)")
ax2.set_xlabel("Projection angle (deg)")
ax2.set_ylabel("Projection position (pixels)")
ax2.imshow(sinogram, cmap=plt.cm.Greys_r,
           extent=(0, 180, 0, sinogram.shape[0]), aspect='auto')

fig.tight_layout()
plt.show()

# Reconstruction with Filtered Back Projection
'''
The mathematical foundation of the filtered back projection is the Fourier slice theorem [2]. It uses Fourier transform of the projection and interpolation in Fourier space to obtain the 2D Fourier transform of the image, which is then inverted to form the reconstructed image. The filtered back projection is among the fastest methods of performing the inverse Radon transform. The only tunable parameter for the FBP is the filter, which is applied to the Fourier transformed projections. It may be used to suppress high frequency noise in the reconstruction. skimage provides a few different options for the filter.
'''
from skimage.transform import iradon

reconstruction_fbp = iradon(sinogram, theta=theta, circle=True)
error = reconstruction_fbp - image
print('FBP rms reconstruction error: %.3g' % np.sqrt(np.mean(error**2)))

imkwargs = dict(vmin=-0.2, vmax=0.2)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5),
                               sharex=True, sharey=True)
ax1.set_title("Reconstruction\nFiltered back projection")
ax1.imshow(reconstruction_fbp, cmap=plt.cm.Greys_r)
ax2.set_title("Reconstruction error\nFiltered back projection")
ax2.imshow(reconstruction_fbp - image, cmap=plt.cm.Greys_r, **imkwargs)
plt.show()
