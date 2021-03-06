'''
The discrete wavelet transform is not shift-invariant. Shift invariance can be achieved through an undecimated wavelet transform (also called stationary wavelet transform), at cost of increased redundancy (i.e. more wavelet coefficients than input image pixels). An alternative way to approximate shift-invariance in the context of image denoising with the discrete wavelet transform is to use the technique known as “cycle spinning”. This involves averaging the results of the following 3-step procedure for multiple spatial shifts, n:

1.) (circularly) shift the signal by an amount, n
2.) apply denoising
3.) apply the inverse shift
'''

import matplotlib.pyplot as plt
from skimage.restoration import denoise_wavelet
from skimage.restoration import cycle_spin # requires skimage v0.14
from skimage import data, img_as_float
from skimage.util import random_noise
from skimage.measure import compare_psnr

original = img_as_float(data.chelsea()[100:250, 50:300])

sigma = 0.155
noisy = random_noise(original, var=sigma**2)

fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(10, 4),
                       sharex=False, sharey=False)
ax = ax.ravel()

psnr_noisy = compare_psnr(original, noisy)
ax[0].imshow(noisy)
ax[0].axis('off')
ax[0].set_title('Noisy\nPSNR={:0.4g}'.format(psnr_noisy))


# Repeat denosing with different amounts of cycle spinning.  e.g.
# max_shift = 0 -> no cycle spinning
# max_shift = 1 -> shifts of (0, 1) along each axis
# max_shift = 3 -> shifts of (0, 1, 2, 3) along each axis
# etc...

denoise_kwargs = dict(multichannel=True, convert2ycbcr=True, wavelet='db1')

all_psnr = []
max_shifts = [0, 1, 3, 5]
for n, s in enumerate(max_shifts):
    im_bayescs = cycle_spin(noisy, func=denoise_wavelet, max_shifts=s,
                            func_kw=denoise_kwargs, multichannel=True)
    ax[n+1].imshow(im_bayescs)
    ax[n+1].axis('off')
    psnr = compare_psnr(original, im_bayescs)
    if s == 0:
        ax[n+1].set_title(
            "Denoised: no cycle shifts\nPSNR={:0.4g}".format(psnr))
    else:
        ax[n+1].set_title(
            "Denoised: {0}x{0} shifts\nPSNR={1:0.4g}".format(s+1, psnr))
    all_psnr.append(psnr)

# plot PSNR as a function of the degree of cycle shifting
ax[5].plot(max_shifts, all_psnr, 'k.-')
ax[5].set_ylabel('PSNR (dB)')
ax[5].set_xlabel('max cycle shift along each axis')
ax[5].grid('on')
plt.subplots_adjust(wspace=0.35, hspace=0.35)

# Annotate with a cyan arrow on the 6x6 case vs. no cycle shift case to
# illustrate a region with reduced block-like artifact with cycle shifting
arrowprops = dict(arrowstyle="simple,tail_width=0.1,head_width=0.5",
                  connectionstyle="arc3",
                  color='c')
for i in [1, 4]:
    ax[i].annotate("", xy=(101, 39), xycoords='data',
                   xytext=(70, 70), textcoords='data',
                   arrowprops=arrowprops)

plt.show()