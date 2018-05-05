'''
In image processing, we frequently apply the same algorithm on a large batch of images. In this paragraph, we propose to use joblib to parallelize loops.

Joblib is a set of tools to provide lightweight pipelining in Python
'''

from skimage import data, color, util
from skimage.restoration import denoise_tv_chambolle
from skimage.feature import hog

def task(image):
    """
    Apply some functions and return an image.
    """
    image = denoise_tv_chambolle(image[0][0], weight=0.1, multichannel=True)
    fd, hog_image = hog(color.rgb2gray(image), orientations=8,
                        pixels_per_cell=(16, 16), cells_per_block=(1, 1),
                        visualize=True)
    return hog_image


# Prepare images
hubble = data.hubble_deep_field()
width = 10
pics = util.view_as_windows(hubble, (width, hubble.shape[1], hubble.shape[2]), step=width)

from joblib import Parallel, delayed
def joblib_loop():
    Parallel(n_jobs=4)(delayed(task)(i) for i in pics)