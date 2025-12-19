# Example of how to use Unnatural_L0 code

import Unnatural_L0
import load_save_image
import kernel_estimation

# Two ways of initialization

# 1. Initialize with blur image and latent image
Y = quick_load_image("1_1_blurred.png")  # blurry image
X = quick_load_image("latent_image.png") # latent image
K = kernel_estimation(Y, X)

# # 2. Initialize with blur image and kernel
# Y = quick_load_image("1_1_blurred.png")  # blurry image
# ker = quick_load_image("kernel1_groundtruth_kernel.png")  

ker_gt /= np.sum(ker_gt)
test = UnnaturalL0UniformLayer(Y, ker_gt)
X, K = test.optimize(2,4,1)

# display and save the result
quick_display(X, None, f"d_{i}_{j}_img") 
quick_display(K, None, f"d_{i}_{j}_ker")
