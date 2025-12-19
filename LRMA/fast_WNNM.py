import numpy as np
from numba import jit
from fast_block_matching import fast_block_matching

@jit(nopython=True)
def process_patch_group(Y_j, sigma_n, c, eps):
    U, sigma_Yj, Vt = np.linalg.svd(Y_j, full_matrices=False)
    n_sigma = len(sigma_Yj)
    sigma_Xj = np.sqrt(np.maximum(0, sigma_Yj**2 - n_sigma*sigma_n**2))
    w_j = c*np.sqrt(n_sigma)/(sigma_Xj + eps)
    S_w = np.maximum(0, sigma_Yj - w_j) 
    X_j = U@np.diag(S_w)@Vt
    return X_j

@jit(nopython=True)
def mean_along_axis1(arr):
    result = np.zeros(arr.shape[0], dtype=arr.dtype)
    for i in range(arr.shape[0]):
        sum_val = 0.0
        for j in range(arr.shape[1]):
            sum_val += arr[i, j]
        result[i] = sum_val / arr.shape[1]
    return result

@jit(nopython=True)
def image_WNNM(image, K=3, sigma_n=19.0):
    n = 8
    delta = 0.1
    c = 2.8
    eps = 1e-6
    
    image_float = image.astype(np.float64)
    
    x = image_float.copy()
    y = image_float.copy()

    for k in range(1, K+1):
        y = x + delta*(image_float - y)

        x_new = np.zeros_like(x, dtype=np.float64)
        x_count = np.zeros_like(x, dtype=np.float64)

        h, w = y.shape
        
        for i in range(0, h-n+1, n-1):
            for j in range(0, w-n+1, n-1):
                curr_i = min(i, h-n)
                curr_j = min(j, w-n)
                
                patch, patchidx, target, taridx = fast_block_matching(y, curr_i, curr_j)

                if patch.shape[1] == 0:
                    continue
                
                Y_j = patch 
                X_j = process_patch_group(Y_j, sigma_n, c, eps)

                Aver_Xj = mean_along_axis1(X_j)
                Aver_Xj = np.reshape(Aver_Xj, (n, n))
                
                x_new[curr_i:curr_i+n, curr_j:curr_j+n] += Aver_Xj
                x_count[curr_i:curr_i+n, curr_j:curr_j+n] += 1.0

        x_new_safe = x_new.copy()
        x_count_safe = x_count.copy()
        
        for i in range(h):
            for j in range(w):
                if x_count_safe[i, j] > 0.0:
                    x[i, j] = x_new_safe[i, j] / x_count_safe[i, j]

    return x

def run_image_WNNM(image, K=1, sigma_n=19.0):
    if image.dtype != np.float64:
        image = image.astype(np.float64)
        
    result = image_WNNM(image, K=K, sigma_n=sigma_n)
    return result
