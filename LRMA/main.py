import cv2
import numpy as np
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import conv
from ELRP import ELRP
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def clip_img(img,ground_img):
    h, w = ground_img.shape
    h_center, w_center = image.shape[0] // 2, image.shape[1] // 2
    h_half, w_half = h // 2, w // 2

    top = h_center - h_half
    bottom = top + h
    left = w_center - w_half
    right = left + w

    return img[top:bottom, left:right]

image = cv2.imread(f"./data/1_1_blurred.png", cv2.IMREAD_GRAYSCALE)
kernel = conv.read_kernel_from_csv(f"./data/recover_image_1_1.csv")
kernel = kernel/np.sum(kernel)
un = cv2.imread(f"./data/recover_image_1_1.jpg", cv2.IMREAD_GRAYSCALE)
un = cv2.resize(un, (image.shape[1], image.shape[0]))
ground_img = cv2.imread(f"./data/img1_groundtruth_img.png", cv2.IMREAD_GRAYSCALE)

b = image
l = un
k = kernel

l = ELRP(l,b,k)

cliped_img = clip_img(l,ground_img)
cliped_un = clip_img(un,ground_img)

val = ssim(cliped_img, ground_img, channel_axis=-1, data_range=cliped_img.max()-cliped_img.min())
print(f"Our SSIM: {val:.2f}")
val = ssim(cliped_un, ground_img, channel_axis=-1, data_range=cliped_un.max()-cliped_un.min())
print(f"Unnatural L0 SSIM: {val:.2f}")

val = psnr(cliped_img, ground_img,data_range=255)
print(f"Our PSNR: {val:.2f} dB")
val = psnr(cliped_un, ground_img,data_range=255)
print(f"Unnatural L0PSNR: {val:.2f} dB")

plt.imshow(l, cmap='gray')
plt.show()





