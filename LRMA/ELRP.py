import numpy as np
from fast_WNNM import run_image_WNNM
from gradient import compute_gradient
from scipy.signal import convolve2d


def solve_d(l,k,b):
    ita = 1
    conv = convolve2d(l, k, mode='same', boundary='fill') - b
    return np.sign(conv) * np.maximum(np.abs(conv) - ita, 0.0)

def solve_l(k,b,d,p,g_x,g_y,beta,tau):
    H, W = b.shape
    h, w = k.shape
    shape = (H + h - 1, W + w - 1)

    k_pad = np.zeros(shape)
    k_pad[shape[0]//2-h//2:shape[0]//2-h//2+h, shape[1]//2-w//2:shape[1]//2-w//2+w] = k

    k_fft = np.fft.fft2(np.fft.fftshift(k_pad))
    
    bd_fft = np.fft.fft2(b+d,s=shape)
    p_fft = np.fft.fft2(p,s=shape)
    g_x_fft = np.fft.fft2(g_x,s=shape)
    g_y_fft = np.fft.fft2(g_y,s=shape)

    kernel_h = np.array([-0.5, 0, 0.5])[None,:]
    kernel_v = np.array([-0.5, 0, 0.5])[:,None]

    Fx = np.fft.fft2(kernel_h, s=shape)
    Fy = np.fft.fft2(kernel_v, s=shape)

    alpha = 1
    beta = 1
    tau = 0.1

    numerator = alpha * np.conj(k_fft) * bd_fft + beta * p_fft + tau * (np.conj(Fx)*g_x_fft + np.conj(Fy)*g_y_fft)
    denominator = alpha * np.conj(k_fft) * k_fft + beta + tau * (np.conj(Fx)*Fx + np.conj(Fy)*Fy) + 1e-6
    l_fft = numerator / denominator
    l = np.real(np.fft.ifft2(l_fft))

    l = l[:H, :W]

    l = np.minimum(l, 255)
    l = np.maximum(l, 0)

    return l

def solve_k(l,b,gamma):
    gamma = 5

    l_fft = np.fft.fft2(l)
    b_fft = np.fft.fft2(b)

    kernel_h = np.array([-0.5, 0, 0.5])[None,:]
    kernel_v = np.array([-0.5, 0, 0.5])[:,None]

    Fx = np.fft.fft2(kernel_h, s=l.shape)
    Fy = np.fft.fft2(kernel_v, s=l.shape)

    l_x_fft = Fx * l_fft
    l_y_fft = Fy * l_fft

    b_x_fft = Fx * b_fft
    b_y_fft = Fy * b_fft

    numerator = np.conj(l_x_fft) * b_x_fft + np.conj(l_y_fft) * b_y_fft
    denominator = np.conj(l_x_fft) * l_x_fft + np.conj(l_y_fft) * l_y_fft + gamma

    k_fft = numerator/denominator
    k = np.real(np.fft.fftshift(np.fft.ifft2(k_fft)))

    
    return k

def ELRP(latent,blur,kernel):
    
    b = blur
    l = latent
    k = kernel

    lambda_1 = 0.05
    sigma = 0.05

    d = solve_d(l,k,b)
    beta = 2*sigma
    p = run_image_WNNM(l,sigma_n=np.sqrt(lambda_1/beta),K=1)
    tau = 2*lambda_1
    l_x,l_y = compute_gradient(l)
    g_x = run_image_WNNM(l_x,sigma_n=np.sqrt(sigma/tau),K=1)
    g_y = run_image_WNNM(l_y,sigma_n=np.sqrt(sigma/tau),K=1)
    l = solve_l(k,b,d,p,g_x,g_y,beta,tau)

    return l

                

