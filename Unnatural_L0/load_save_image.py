import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

from numpy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.signal import convolve2d, correlate2d
from scipy.ndimage import gaussian_filter

def quick_load_image(fpath):
    im = cv.imread(fpath, cv.IMREAD_GRAYSCALE)
    if im is None:
        raise FileNotFoundError(f"Error: Image file '{fpath}' not found.")
    return im.astype(np.float32) / 255

def quick_display(im, title=None, fn=None):
    # a = np.min(im)
    # b = np.max(im)
    # im = (im-a)/(b-a)
    im = np.real(im)

    plt.imshow(im, cmap='grey', clim=(0,1))
    plt.axis('off')  # Remove axes for better visualization
    if title:
        plt.title(title)
    if fn is not None:
        plt.imsave(fn+".png",im, cmap="grey")
    plt.show()

def range_re_im(arr):
    a = np.min(np.real(arr))
    b = np.max(np.real(arr))
    c = np.min(np.imag(arr))
    d = np.max(np.imag(arr))
    print("re", a,b,"\nim:",c,d)

def crop_center(img, shape):
    h, w = img.shape
    ch, cw = h // 2, w // 2
    kh, kw = shape
    return img[ch - kh//2 : ch + kh//2, cw - kw//2 : cw + kw//2]


def FFT_conv2D(img, ker, corr=False):
    rows, cols = img.shape
    hrows, hcols = ker.shape

    if corr:
        ker = np.flip(ker, axis=(0,1))

    # Circularly pad the image
    i_padded = np.pad(img, [(hrows - 1, hrows - 1), (hcols - 1, hcols - 1)], mode='wrap')
    pad_rows, pad_cols = i_padded.shape

    # Zero-pad the kernel to match padded image size and shift center
    h_padded = np.zeros((pad_rows, pad_cols))
    h_padded[:hrows, :hcols] = ker
    h_padded = np.roll(h_padded, shift=(-hrows // 2, -hcols // 2), axis=(0, 1))

    # Fourier transforms
    ipadded_ft = fft2(i_padded)
    hpadded_ft = fft2(h_padded)

    # Element-wise multiplication in frequency domain and inverse FFT
    conv_image_fourier = np.real(ifft2(ipadded_ft * hpadded_ft))

    # Crop back to original image size
    frows, fcols = conv_image_fourier.shape
    rs = (frows - rows) // 2
    cs = (fcols - cols) // 2
    return conv_image_fourier[rs:rs + rows, cs:cs + cols]
