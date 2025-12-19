import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

from numpy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.signal import convolve2d, correlate2d
from scipy.ndimage import gaussian_filter

class ImagePyramid():
    def __init__(self, img, num_level=4):
        self.img = img.astype(np.float32)
        self.num_level = num_level
        self.GaussianPyramid = [img]
        self.LaplacianPyramid = []
        self.reconstructed = None
        self.generate_pyramids()

    def pyramid_downsample(self, img):
        blurred = gaussian_filter(img, sigma=1)
        return blurred[::2, ::2]

    def pyramid_upsample(self, img, target_shape):
        up = np.zeros((img.shape[0]*2, img.shape[1]*2), dtype=np.float32)
        up[::2, ::2] = img
        up = gaussian_filter(up, sigma=1) * 4
        return up[:target_shape[0], :target_shape[1]]

    def generate_pyramids(self):
        current = self.img
        for _ in range(self.num_level - 1):
            current = self.pyramid_downsample(current)
            self.GaussianPyramid.append(current)

        for i in range(self.num_level - 1):
            upsampled = self.pyramid_upsample(self.GaussianPyramid[i+1], self.GaussianPyramid[i].shape)
            lap = self.GaussianPyramid[i] - upsampled
            self.LaplacianPyramid.append(lap)
        self.LaplacianPyramid.append(self.GaussianPyramid[-1])

    def reconstruct(self):
        image = self.LaplacianPyramid[-1]
        for i in reversed(range(self.num_level - 1)):
            image = self.pyramid_upsample(image, self.LaplacianPyramid[i].shape)
            image += self.LaplacianPyramid[i]
        self.reconstructed = image
        return image

    def fusion(self, mask, ref, alpha=0.5):
        fused_pyramid = []
        for i in range(self.num_level):
            l1 = self.LaplacianPyramid[i]
            l2 = ref.LaplacianPyramid[i]
            m = mask if mask.ndim == 2 else mask[:, :, 0]
            m = m.astype(np.float32)
            m = np.clip(m, 0, 1)
            fused = alpha * l1 * m + (1 - alpha) * l2 * (1 - m)
            fused_pyramid.append(fused)

        img = fused_pyramid[-1]
        for i in reversed(range(self.num_level - 1)):
            img = self.pyramid_upsample(img, fused_pyramid[i].shape)
            img += fused_pyramid[i]
        return img

    def display_Gaussian(self):
      for i in range(self.num_level):
        quick_display(self.GaussianPyramid[i])

    def display_Laplacian(self):
      for i in range(self.num_level):
        quick_display(self.LaplacianPyramid[i])
