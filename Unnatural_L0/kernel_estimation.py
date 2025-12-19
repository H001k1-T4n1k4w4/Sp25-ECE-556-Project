import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift

def grad(im, axis=0):
    g = np.zeros(im.shape)
    if axis == 0:
        g[:-1, :] = im[1:, :] - im[:-1, :]
        g[-1, :] = -g[-2, :]
    elif axis == 1:
        g[:, :-1] = im[:, 1:] - im[:, :-1]
        g[:, -1] = -g[:, -2]
    return g

def kernel_estimation(blur_img, sharp_img, ker_size=(31,31)):
    """
    Estimate the blur kernel of single channel 2D image.
    Input: 2D blur image, 2D sharp image, kernel size (optinal)
    Output: blur kernel
    """
    dIx = fft2(grad(sharp_img,0))
    dIy = fft2(grad(sharp_img,1))
    dBx = fft2(grad(blur_img,0))
    dBy = fft2(grad(blur_img,1))
    k = np.abs(ifftshift(ifft2((np.conj(dIx)*dBx+np.conj(dIy)*dBy)/(np.conj(dIx)*dIx+np.conj(dIy)*dIy+5))))
    h, w = blur_img.shape
    kh, kw = ker_size
    h = h // 2 - kh // 2 -1
    w = w // 2 - kw // 2 -1
    return k[h:h+kh, w:w+kw]

