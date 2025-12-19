import numpy as np
# import matplotlib.pyplot as plt
# import cv2 as cv

from numpy.fft import fft2, ifft2, fftshift, ifftshift
# from scipy.signal import convolve2d, correlate2d
from scipy.ndimage import gaussian_filter

class UnnaturalL0UniformLayer():
    def __init__(self, Y, ker=None, ker_shape=(31,31), eps=1, lambd=2e-3, gamma=40):
        """
        Initializes the deblurring layer.
        Y: Input blurred image (grayscale, normalized to [0,1])
        ker_shape: Kernel shape (height, width)
        eps: Initial threshold for gradient thresholding (GNC parameter)
        lambd: Regularization weight for latent image update
        gamma: Regularization weight for kernel update
        """
        self.img_shape = Y.shape
        if ker is not None:
            self.ker_shape = ker.shape
        else:
            self.ker_shape = ker_shape
        self.eps = eps
        self.lambd = lambd
        self.gamma = gamma

        # Resize kernel if larger than image dimensions.
        if self.ker_shape[0] > self.img_shape[0]:
            print("kernel height > image height, resizing.")
            self.ker_shape = (self.img_shape[0], self.ker_shape[1])
        if self.ker_shape[1] > self.img_shape[1]:
            print("kernel width > image width, resizing.")
            self.ker_shape = (self.ker_shape[0], self.img_shape[1])

        self.Y = np.copy(Y)
        self.Y_fft = fft2(self.Y)
        self.X = gaussian_filter(Y, sigma=1)
        # self.X = np.copy(Y)

        # just for tracking
        self.h, self.w = self.Y.shape
        self.N = self.h * self.w

        # Auxiliary variable for gradient thresholding (for vertical and horizontal directions)
        self.l = np.zeros((2, self.h, self.w))

        # Initialize the kernel as a delta function.
        if ker is not None:
            self.K = ker
        else:
            self.init_estimation()
    
    def grad(self, z, axis=0):
        """
        Computes the finite difference gradient along the specified axis.
        Uses backward differences at the boundary.
        """
        g = np.zeros(self.img_shape)
        if axis == 0:
            g[:-1, :] = z[1:, :] - z[:-1, :]
            g[-1, :] = -g[-2, :]
        elif axis == 1:
            g[:, :-1] = z[:, 1:] - z[:, :-1]
            g[:, -1] = -g[:, -2]
        return g/2

    def shock_filter(self, im):
        Ix = self.grad(im, 0)
        Ixx = self.grad(Ix, 0)
        Iy = self.grad(im, 1)
        Iyy = self.grad(Iy, 1)
        Ixy = self.grad(Ix,1)
        return -(np.sign(Ix*Ix*Ixx+2*Ix*Iy*Ixy+Iy*Iy*Iyy) * np.sqrt(Ix**2+Iy**2))

    def init_estimation(self, im=None, ker_size=None, itr = 2, gam=1, lam=2e-3):
        if im is None:
            im = self.Y
        if ker_size is None:
            ker_size = self.ker_shape
        # gradient of blur img
        dBx = fft2(self.grad(im,0))
        dBy = fft2(self.grad(im,1))
        # Pad the image X for patch extraction
        H, W = im.shape
        kH , kW = ker_size
        pad_H = kH // 2
        pad_W = kW // 2
        #  Eq 2: Compute r
        # Compute gradient of the blurred image
        Bx = self.grad(im, 0)
        By = self.grad(im, 1)
        I_edge = np.sqrt(Bx**2 + By**2)

        # Pad the gradients for patch extraction
        Bx_pad = np.pad(Bx, ((pad_H, pad_H), (pad_W, pad_W)), mode='reflect')
        By_pad = np.pad(By, ((pad_H, pad_H), (pad_W, pad_W)), mode='reflect')
        I_edge_pad = np.pad(I_edge, ((pad_H, pad_H), (pad_W, pad_W)), mode='reflect')

        # Eq 2: Compute r
        r = np.zeros_like(im)
        for i in range(H):
            for j in range(W):
                gx_patch = Bx_pad[i:i+kH, j:j+kW]
                gy_patch = By_pad[i:i+kH, j:j+kW]

                sum_vec = np.array([gx_patch.sum(), gy_patch.sum()])
                a = np.linalg.norm(sum_vec)  # Numerator: vector magnitude
                b = np.sum(I_edge_pad[i:i+kH, j:j+kW])  # Denominator: sum of magnitudes
                r[i, j] = a / (b + 0.5)

        quick_display(I_edge, "Blurred gradient magnitude")
        # Calaulate the thresholds tr, ts
        ntr = int(0.5*np.sqrt(kH*kW*H*W))
        nts = int(2*np.sqrt(kH*kW))
        flat_r = r.flatten()
        tr = flat_r[np.argsort(flat_r)[-ntr]]
        It = self.shock_filter(gaussian_filter(im,1))
        Itx = self.grad(It,0)
        Ity = self.grad(It,1)
        It_edge = np.sqrt(Itx**2+Ity**2)
        M = np.where(r>tr, 1, 0)
        MIt = np.where(M, It_edge, 0)
        MIt_flat = MIt.flatten()
        ts = MIt_flat[np.argsort(MIt_flat)[-nts]]

        quick_display(It, "shock filtered")
        quick_display(It_edge, "shock filtered edge")

        # debug, ker index
        mh = H // 2 - kH // 2
        mw = W // 2 - kW // 2

        # Loop through Eq 4, 6, 8
        for i in range(itr):
            print(f"Kernel initializing: itr:{i}")
            # Eq 4
            It = self.shock_filter(gaussian_filter(im,1))
            Itx = self.grad(It,0)
            Ity = self.grad(It,1)
            It_edge = np.sqrt(Itx**2+Ity**2)
            MIt = np.where(M, It_edge, 0)
            M2 = np.where(MIt>ts, 1, 0)
            Isx = np.where(M2, Itx, 0)
            Isy = np.where(M2, Ity, 0)
            # Eq 6
            dIx = fft2(Isx)
            dIy = fft2(Isy)
            K = (np.conj(dIx)*dBx+np.conj(dIy)*dBy)/(np.conj(dIx)*dIx+np.conj(dIy)*dIy+gam)
            k = np.abs(ifftshift(ifft2(K)))
            self.K = k[mh:mh+kH, mw:mw+kW]
            # Eq 8
            im = self.deblur()
            # update thresholds
            ts /= 1.1
            tr /= 1.1
            quick_display(self.K)
        self.X = self.deblur()
        return self.X , self.K

    def phi(self, z, axis=0, eps=None):
        """
        Computes the piecewise sparsity penalty.
        For |grad| <= eps, returns (grad^2)/(eps^2), otherwise returns 1.
        """
        if eps is None:
            eps = self.eps
        g = np.abs(self.grad(z, axis=axis))
        return np.where(g <= eps, (g ** 2) / (eps ** 2), 1.0)

    def phi_0(self, z, axis=0, eps=None):
        """
        Computes the sum of sparsity penalties over the image.
        """
        if eps is None:
            eps = self.eps
        return np.sum(self.phi(z, axis=axis, eps=eps))

    def update_l(self, im = None, eps=None):
        """
        Updates the auxiliary variable 'l' using hard thresholding.
        Uses the threshold parameter eps.
        """
        if eps is None:
            eps = self.eps
        if im is None:
            im = self.X
        for axis in [0, 1]:
            g = self.grad(im, axis=axis)
            # range_re_im(g)
            # tmp = np.where(np.abs(g) <= eps, 0, g)
            # quick_display(tmp)
            # range_re_im(tmp)
            self.l[axis] = g

    def update_x(self, lam=None, eps=None):
        """
        Updates the latent image X using FFT-based deconvolution.
        lam: Regularization parameter.
        eps: Threshold parameter for the current iteration.
        """
        if lam is None:
            lam = self.lambd
        if eps is None:
            eps = self.eps

        c = lam / (eps ** 2)

        # Kernel centering for FFT
        K_padded = np.zeros_like(self.Y)
        kh, kw = self.K.shape
        K_padded[:kh, :kw] = self.K
        K_padded = np.roll(K_padded, -kh // 2, axis=0)
        K_padded = np.roll(K_padded, -kw // 2, axis=1)
        K_fft = fft2(K_padded)

        # Reshape auxiliary gradient variables.
        lh = self.l[1]
        lv = self.l[0]

        # Derivative filters (horizontal and vertical finite differences)
        Dh = np.zeros_like(self.Y)
        Dv = np.zeros_like(self.Y)
        Dh[0, -1] = 0.5
        Dh[0, 0] = -0.5
        Dv[-1, 0] = 0.5
        Dv[0, 0] = -0.5

        # Compute FFTs of derivative filters and gradients.
        Dh_fft = fft2(Dh)
        Dv_fft = fft2(Dv)
        lh_fft = fft2(lh)
        lv_fft = fft2(lv)
        Y_fft = fft2(self.Y)

        # Compute numerator and denominator in the Fourier domain.
        numerator = np.conj(K_fft) * Y_fft + c * (np.conj(Dh_fft) * lh_fft + np.conj(Dv_fft) * lv_fft)
        denominator = (np.conj(K_fft) * K_fft) + c * (np.conj(Dh_fft) * Dh_fft + np.conj(Dv_fft) * Dv_fft)
        # denominator = np.maximum(denominator, 1e-6)

        # Solve for X in the Fourier domain.
        x_fft = numerator / denominator
        X = ifft2(x_fft)
        self.X = np.real(X)
        # self.X = np.clip(self.X, 0, 1)
        return self.X

    def update_k(self, blur_img=None, sharp_img=None, gam = None, ker_size=None):
        """
        Estimate the blur kernel of single channel 2D image.
        Input: 2D blur image, 2D sharp image, kernel size (optinal)
        Output: blur kernel
        """
        if blur_img is None:
            blur_img = self.Y

        if sharp_img is None:
            sharp_img = self.X

        if gam is None:
            gam = 1

        if ker_size is None:
            ker_size = self.ker_shape

        dIx = fft2(self.grad(sharp_img,0))
        dIy = fft2(self.grad(sharp_img,1))
        dBx = fft2(self.grad(blur_img,0))
        dBy = fft2(self.grad(blur_img,1))
        k = np.abs(ifftshift(ifft2((np.conj(dIx)*dBx+np.conj(dIy)*dBy)/(np.conj(dIx)*dIx+np.conj(dIy)*dIy+gam))))
        h, w = blur_img.shape
        kh, kw = ker_size
        h = h // 2 - kh // 2 -1
        w = w // 2 - kw // 2 -1
        self.K = k[h:h+kh, w:w+kw]
        return self.K
    
    def optimize(self, t=5, itr=4, eps=None):
        """
        Main optimizing loop with graduated non-convexity.
        t: Number of outer iterations.
        itr: Number of inner iterations per outer iteration.
        eps: Initial threshold value (if not provided, uses self.eps).
        """
        if eps is None:
            eps = self.eps
        for ti in range(t):
            epsilon = eps
            for i in range(itr):
                # print(self.X.shape)
                num_iter = min(int(1 / epsilon), 8)
                for j in range(num_iter):
                    self.update_l(eps=epsilon)  # Use the current epsilon for thresholding.
                    self.update_x(eps=epsilon)
                epsilon /= 2  # Gradually reduce epsilon.
                # print(f"Update Image...{i}")
            # self.update_k(gam=100)
            # print(f"Update Kernel...{ti}")
            # quick_display(self.X,f"X{ti}")
            # quick_display(self.K,f"K{ti}")
        return self.X, self.K
    
    def deblur(self, eps=0.1):
        self.update_l(eps=eps)  # Use the current epsilon for thresholding.
        self.update_x(eps=eps)
        return self.X
