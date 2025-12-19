import numpy as np
import cv2

def compute_gradient(image):
    """
    Compute the gradients of an image in x and y directions.
    
    Args:
        image: Input image (grayscale) as a numpy array
        
    Returns:
        gradient_x: Gradient in the x direction (horizontal)
        gradient_y: Gradient in the y direction (vertical)
    """

    image = image.astype(np.float32)

    gradient_x = np.zeros_like(image, dtype=np.float32)
    gradient_x[:, 1:-1] = (image[:, 2:] - image[:, :-2]) / 2.0
    

    gradient_x[:, 0] = image[:, 1] - image[:, 0]
    gradient_x[:, -1] = image[:, -1] - image[:, -2]
    
    gradient_y = np.zeros_like(image, dtype=np.float32)
    gradient_y[1:-1, :] = (image[2:, :] - image[:-2, :]) / 2.0
    
    gradient_y[0, :] = image[1, :] - image[0, :]
    gradient_y[-1, :] = image[-1, :] - image[-2, :]
    
    return gradient_x, gradient_y


def compute_gradient_cv2(image):
    """
    Compute the gradients of an image in x and y directions using OpenCV.
    
    Args:
        image: Input image (grayscale) as a numpy array

    Returns:
        gradient_x: Gradient in the x direction (horizontal)
        gradient_y: Gradient in the y direction (vertical)
    """

    image = image.astype(np.float32)
    
    gradient_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    
    return gradient_x, gradient_y


