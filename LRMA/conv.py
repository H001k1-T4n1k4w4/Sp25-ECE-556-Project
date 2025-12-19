import numpy as np
import csv
import os
import cv2  

def read_kernel_from_csv(csv_file_path):
    """
    Read convolution kernel data from a CSV file
    
    Args:
        csv_file_path: Path to the CSV file containing the kernel
        
    Returns:
        kernel: NumPy array containing kernel values
    """
    try:
        # Check if file exists
        if not os.path.exists(csv_file_path):
            raise FileNotFoundError(f"CSV file not found: {csv_file_path}")
        
        # Read CSV file
        kernel_data = []
        with open(csv_file_path, 'r', newline='') as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                # Convert string values to float
                float_row = [float(val) for val in row if val.strip()]
                if float_row:  # Only add non-empty rows
                    kernel_data.append(float_row)
        
        # Convert to NumPy array
        kernel = np.array(kernel_data, dtype=np.float32)
        
        # Check if kernel is empty
        if kernel.size == 0:
            raise ValueError("Kernel is empty. Check if CSV file contains valid data.")
        
        return kernel
    
    except Exception as e:
        print(f"Error reading kernel from CSV: {e}")
        return None

def apply_convolution(image, kernel):
    """
    Apply convolution to an image using the given kernel
    
    Args:
        image: Input image as numpy array
        kernel: Convolution kernel as numpy array
        
    Returns:
        result: Convolution result as numpy array
    """
    # Input validation
    if image is None or kernel is None:
        print("Error: Invalid input for convolution")
        return None
    
    # Convert image to float32 for better precision
    if image.dtype != np.float32:
        image_float = image.astype(np.float32)
    else:
        image_float = image.copy()
    
    # Apply convolution using OpenCV's filter2D function
    # -1 means the output has the same depth as the input
    result = cv2.filter2D(image_float, -1, kernel)
    
    return result

def wiener_deconvolution(image, kernel, K=0.01):
    """
    Perform deconvolution (inverse convolution) using Wiener filtering

    Args:
    - image: Input blurred image (numpy array)
    - kernel: Degradation convolution kernel (numpy array)
    - K: Parameter for Wiener filter (default 0.01)

    Returns:
    - Processed deblurred image
    """
    # Calculate the Fourier transform of the image
    image_fft = np.fft.fft2(image)
    kernel_fft = np.fft.fft2(kernel, s=image.shape)

    # Calculate the Wiener deconvolution formula: H^* / ( |H|^2 + K )
    kernel_fft_conj = np.conj(kernel_fft)
    kernel_fft_abs2 = np.abs(kernel_fft) ** 2

    # Wiener filter formula
    deconvolved_fft = (kernel_fft_conj / (kernel_fft_abs2 + K)) * image_fft
    deconvolved_image = np.fft.ifft2(deconvolved_fft).real

    # Normalize to 0-255
    deconvolved_image = np.clip(deconvolved_image, 0, 255).astype(np.uint8)

    return deconvolved_image

