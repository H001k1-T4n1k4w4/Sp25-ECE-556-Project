# ECE 556 Project

This repository is the implementation of our ECE 556 project by Puyu He, 
Rui-Yu Lin, https://github.com/RUI030,
Xiangzhong Ye, https://github.com/Suckl.

Our project goal is to reproduce the result by Wenqi Ren et al. [“Image deblurring via enhanced low-rank prior”](https://ieeexplore.ieee.org/abstract/document/7473901).

## Usage

### LRMA (main part)
Python implementation of algorithm proposed by Ren et al.
Run main.py to see an example of image deblurring.

```
python LRMA/main.py
```

After running main.py, the deblurred picture will be shown and metrics will also be reported. For result of other pictures, one need to modify variables `image`\(blurred image\) , `ground_img` (ground truth image), `kernel` and `un`\(output of unnatural L0\) .

For other files, 
* `ELRP.py` is the python implementation of the proposed algorithm.
* `fast_WNNM.py` is the implementation of the algorithm proposed by Gu et al. to solve the WNNM problem.
* `fast_block_matching.py` utilize numba to accelerate the python code of block matching.
  
### Block Matching (standalone function)
Python reimplementaton of C based block matching funtion of the [An Analysis and Implementation of the BM3D Image Denoising Method](https://www.ipol.im/pub/art/2012/l-bm3d/) by Marc Lebrun. Following is an example of calling the precompute_BM().

```python
import BM as BM

# Load image
PATH_IMAGE = 'img1_groundtruth_img.png'
img = cv.imread(PATH_IMAGE, cv.IMREAD_GRAYSCALE)

# Hyper parameters
kHW = 4         # Half width of patches
NHW = 16        # Maximum number of similar patches
nHW = 15        # Half width of search window
pHW = 150       # Step between processed patches
tauMatch = 1    # Distance threshold

# Run block matching
patches3d, patches2d = BM.precompute_BM(img, kHW, NHW, nHW, pHW, tauMatch)
```
Returns：  
    patches3d：numpy.ndarray with shape (NHW * # of reference patch, 2 * kHW, 2 * kHW)
<br/>
    patches2d：numpy.ndarray with shape (NHW * # of reference patch, (4 * kHW **2))
    
### Unnatural L0
Reimplementaton of [Unnatural L0 Sparse Representation for Natural Image Deblurring](https://ieeexplore.ieee.org/document/6618991) by Xu _et al._
This part contains single layer of the Unnatural L0 deblurring class, which can be applied in layers of image pyramid. Following is an example of how to use the class. 
Additionally, it provides early stage estimation of the blur kernel in Ren et al's method. <br/>
Note: We found that this method heavily rely on properly initialized kernel

```python
from Unnatural_L0 import Unnatural_L0

Y = # blur image, 2D numpy array
K_init = # initialized kernel, 2D numpy array
layer = UnnaturalL0UniformLayer(Y, K_init)
X, K = layer.optimize() # deblur and return the latent image and updated kernel
```
or optimize with specified the parameters
```python
X, K = layer.optimize(num_iter, number_iter_image, init_epsilon)
```
In each iteration, $\epsilon$ is initialized to `init_epsilon`, and the image is updated `number_iter_image` times with epsilon divided by 2 after each image update, then kernel is updated once.
<br/>
For other files, 
* `load_save_image.py` provide functions for quickly load, display and save image.
* `kernel_estimation.py` provide functions for estimating kernel given blurred image and latent image.
* `image_pyramid.py` provide functions for quickly constructed an image pyramid




>📋  Pick a licence and describe how to contribute to your code repository. 
