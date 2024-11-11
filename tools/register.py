import matplotlib.pyplot as plt
import numpy as np
import scipy
import tools.math
import tools.display
import matplotlib.patches as patches
from scipy import interpolate
from scipy import ndimage
import math
import random
 
def check_stable_ssd(history, n_last, max_std):
    """
    Check if the SSD values have stabilized over the last 'n_last' iterations.
   
    Parameters:
    - history: List of SSD values through iterations
    - n_last: Number of last iterations to check for stabilization
    - max_std: Maximum standard deviation threshold for considering stabilization
   
    Returns:
    - True if SSD values have stabilized (i.e., standard deviation < max_std), False otherwise.
    """
    if len(history) < n_last:
        return False  # Not enough elements to check for stability
 
    # Ensure the last SSD value is less than the previous one (indicating improvement)
    if history[-1] > history[-2]:
        return False  # SSD should decrease or remain stable over iterations
 
    # Extract the last 'n_last' SSD values
    last_values = history[-n_last:]
   
    # Compute the standard deviation of the last 'n_last' values
    std = np.std(last_values)
   
    # Return True if standard deviation is below the threshold, indicating stabilization
    return std < max_std
   
def translate_image(I, p, q):
    """
    Translates the input image by specified horizontal and vertical offsets.
 
    Parameters:
    - I (np.array): The input image to be translated, represented as a 2D NumPy array.
    - p (float): The horizontal translation offset (positive for rightward shift).
    - q (float): The vertical translation offset (positive for downward shift).
 
    Returns:
    - np.array: The translated image, as a NumPy array.
    """
   
    # Get the original image dimensions
    h, w = I.shape
 
    # Create a grid of x and y coordinates
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
 
    # Apply translation offsets
    x_translated = x + p
    y_translated = y + q
 
    # Interpolate using map_coordinates
    img_t = ndimage.map_coordinates(I, [y_translated.ravel(), x_translated.ravel()], order=5).reshape(h, w)
    img_t = np.clip(img_t, 0, 255).astype(np.uint8)
    return img_t
 
def register_translation_ssd(
    img_1, img_2, eta, n_iter, conv_thresh, n_last, max_std):
    """
    Register two images by adjusting translation to minimize SSD.
   
    Parameters:
    - img_1: Reference image (numpy array)
    - img_2: Image to be translated (numpy array)
    - eta: Step size for gradient descent on translation (default: 1e-6)
    - n_iter: Maximum number of iterations
    - conv_thresh: Threshold for stopping based on SSD
    - n_last: Number of elements to check for unchanged percentages
    - max_std: Maximum standard deviation threshold for SSD stability check
   
    Returns:
    - reg_images: List of translated images through iterations
    - ssd_hist: List of SSD values through iterations
    - p_hist: List of horizontal translations through iterations
    - q_hist: List of vertical translations through iterations
    """
    # Initialize storage for images and SSD values
    reg_images = []
    ssd_hist = []
   
    # Initialize translation parameters
    p = 0  # Horizontal translation
    q = 0  # Vertical translation
    p_hist = []
    q_hist = []
   
    for iter_count in range(n_iter):
        p_hist.append(p)
        q_hist.append(q)
       
        # Apply current translation to the image
        new_img_2 = translate_image(img_2, p, q)
        reg_images.append(new_img_2)
       
        # Calculate the new SSD value
        new_ssd = tools.math.ssd(img_1, new_img_2)
        ssd_hist.append(new_ssd)
       
        # Check stopping criteria
        if new_ssd <= conv_thresh or check_stable_ssd(ssd_hist, n_last, max_std):
            break
       
        # Compute gradients with respect to horizontal and vertical translations
        grad_p = np.sum((new_img_2 - img_1) * np.gradient(new_img_2, axis=1))  # x-axis gradient
        grad_q = np.sum((new_img_2 - img_1) * np.gradient(new_img_2, axis=0))  # y-axis gradient
       
        # Update p and q based on gradients
        p -= eta * grad_p
        q -= eta * grad_q
       
        # Log current iteration details
        print(
            f"{'Iter':<5}{iter_count:<3} | "
            f"{'SSD':<5}{round(ssd_hist[-1], 2):<12} | "
            f"{'%SSD(aft/bef)':<18}{round(ssd_hist[-1] * 100 / ssd_hist[-2], 3) if len(ssd_hist) >= 2 else 'None':<7} | "
            f"{'Grad p':<8}{round(grad_p, 3):<10} | "
            f"{'P':<5}{round(p, 2):<8} | "
            f"{'Grad q':<8}{round(grad_q, 3):<10} | "
            f"{'Q':<5}{round(q, 2):<8}"
        )
 
    return reg_images, ssd_hist, p_hist, q_hist
 
def rotate_image(I, theta):
    """
    Rotates the input image by a specified angle.
 
    Parameters:
    - I (np.array): The input image to be rotated, represented as a 2D NumPy array.
    - theta (float): The rotation angle in radians, counterclockwise.
 
    Returns:
    - np.array: The rotated image, as a NumPy array.
    - list: A list containing the x and y coordinates of the rotated grid.
    """
   
    # Get the original image dimensions
    h, w = I.shape
 
    # Create a grid of x and y coordinates
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
 
    # Apply rotation transformation
    x_rotated = x * np.cos(theta) - y * np.sin(theta)
    y_rotated = x * np.sin(theta) + y * np.cos(theta)
 
    # Use map_coordinates with rotated coordinates
    img_r = ndimage.map_coordinates(I, [y_rotated.ravel(), x_rotated.ravel()], order=5).reshape(h, w)
    img_r = np.clip(img_r, 0, 255).astype(np.uint8)
    return img_r, [x_rotated, y_rotated]
 
def register_rotation_ssd(
    img_1, img_2, eta_r, n_iter, conv_thresh, n_last, max_std):
    """
    Register two images by adjusting rotation to minimize SSD using gradient descent.
   
    Parameters:
    - img_1: Reference image (numpy array)
    - img_2: Image to be rotated (numpy array)
    - eta_r: Learning rate for gradient descent on rotation angle (default: 1e-10)
    - n_iter: Maximum number of iterations
    - conv_thresh: Threshold for stopping based on SSD
    - n_last: Number of elements to check for unchanged percentages
    - max_std: Maximum standard deviation threshold for SSD stability check
   
    Returns:
    - reg_images: List of rotated images through iterations
    - ssd_hist: List of SSD values through iterations
    - theta_hist: List of rotation angles (theta) through iterations
    """
    # Storage for transformed images, SSD values, and rotation angles
    reg_images = []
    ssd_hist = []
    theta_hist = []
 
    # Initialize rotation angle (starting small)
    theta = 0  # Initial small rotation angle (1 degree)
 
    # Iterate for a fixed number of iterations
    for iter_count in range(n_iter):
        theta_hist.append(theta)
       
        # Apply rotation to the image
        new_img_2, coords = rotate_image(img_2, theta)
        reg_images.append(new_img_2)
       
        # Calculate new SSD and add to history
        new_ssd = tools.math.ssd(img_1, new_img_2)
        ssd_hist.append(new_ssd)
       
        # Stop if SSD is below convergence threshold/ is stable over the last n iterations (check using max_std)
        if ssd_hist[-1] <= conv_thresh or check_stable_ssd(ssd_hist, n_last, max_std):
            break
       
        # Compute gradient with respect to rotation
        x_coords, y_coords = coords[0], coords[1]
        grad_y, grad_x = np.gradient(new_img_2)
        grad_t = np.sum((new_img_2 - img_1) * (-grad_x * y_coords + grad_y * x_coords))
 
        # Update rotation angle using gradient descent
        theta -= eta_r * grad_t
 
        # Print iteration status
        print(
            f"{'Iter':<5}{iter_count + 1:<3} | "
            f"{'SSD':<5}{round(ssd_hist[-1], 2):<10} | "
            f"{'%SSD(aft/bef)':<18}{round(ssd_hist[-1] * 100 / ssd_hist[-2], 3) if len(ssd_hist) >= 2 else 'None':<7} | "
            f"{'Grad t':<8}{round(grad_t, 3):<10} | "
            f"{'Theta':<6}{round(theta, 3):<6}"
        )
 
    return reg_images, ssd_hist, theta_hist
 
def rigid_image(I, p, q, theta):
    """
    Applies rigid transformation (rotation + translation) to an input image.
 
    Parameters:
    - I (np.array): The input image to be transformed, represented as a 2D NumPy array.
    - p (float): The horizontal translation offset (positive for rightward shift).
    - q (float): The vertical translation offset (positive for downward shift).
    - theta (float): The angle by which the image is rotated, in radians (counter-clockwise).
 
    Returns:
    - np.array: The transformed image, as a NumPy array.
    - list: The rotated coordinates [x_rotated, y_rotated].
    """
   
    # Get the original image dimensions
    h, w = I.shape
 
    # Create a grid of x and y coordinates
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
 
    # Apply rotation transformation
    x_rotated = x * np.cos(theta) - y * np.sin(theta)
    y_rotated = x * np.sin(theta) + y * np.cos(theta)
   
    x_translated = x_rotated + p
    y_translated = y_rotated + q
 
    # Use map_coordinates with rotated coordinates
    img_r = ndimage.map_coordinates(I, [y_translated.ravel(), x_translated.ravel()], order=5).reshape(h, w)
    img_r = np.clip(img_r, 0, 255).astype(np.uint8)
    return img_r, [x_rotated, y_rotated]
 
def register_rigid_ssd(
    orig_img1, orig_img2, grad_optimizer, orig_eta_t, orig_eta_r, n_iter, conv_thresh,
    res_level, gauss_sigma, n_last, max_std, momentum=0.9
):
    """
    Perform rigid image registration using SSD (Sum of Squared Differences) as the similarity measure.
    This function implements both regular gradient descent and momentum-based optimization methods for image alignment.
    The registration is performed across multiple scales, allowing for better performance at different resolutions.
 
    Parameters:
    - orig_img1 (numpy array): The reference image that will be used as the target for registration.
    - orig_img2 (numpy array): The image to be registered to the reference image (orig_img1).
    - grad_optimizer (int): The type of gradient descent optimizer:
                             1 for regular gradient descent, other values for momentum-based optimization.
    - orig_eta_t (float): Learning rate for translation updates.
    - orig_eta_r (float): Learning rate for rotation updates.
    - n_iter (int): The number of iterations to run the optimization loop.
    - conv_thresh (float): Threshold for convergence based on SSD value.
    - res_level (int): The number of resolution levels (scales) to perform the multi-scale registration.
    - gauss_sigma (float): Standard deviation for Gaussian smoothing, used for image blurring during registration.
    - n_last (int): Number of the last SSD values to check for stability.
    - max_std (float): Maximum standard deviation of the last n_last SSD values to check for convergence.
    - momentum (float, optional): The momentum factor for velocity updates in momentum-based optimization (default is 0.9).
 
    Returns:
    - reg_images (list): List of registered images at each resolution level.
    - ssd_hist_all_scales (list): List of SSD values across all scales.
    - p_hist (list): History of translation parameter p (x-axis).
    - q_hist (list): History of translation parameter q (y-axis).
    - theta_hist (list): History of rotation parameter theta.
    - scale_hist (list): History of resolution levels (scales) used during registration.
    """
    # Initialize transformation parameters
    reg_images = []  # Store registered images for each scale
    ssd_hist_all_scales = []
    p, q, theta = 0, 0, 0  # Initial translations and rotation
 
    # Store parameter histories
    p_hist = []  # History of p values
    q_hist = []  # History of q values
    theta_hist = []  # History of theta values
 
    # Set scaling factors for multiscale processing
    scales = [2**s for s in range(res_level + 1)][::-1]
    scale_hist = []
 
    # For ADAM
    beta_1, beta_2 = 0.9, 0.99
    epsilon = 1e-8
   
    for s, scale in enumerate(scales):
       
        ssd_hist = []  # Store SSD values at this scale
 
        # Adjust translations and rotation for this scale
        if s > 0:
            p = scales[s - 1] * p / scale
            q = scales[s - 1] * q / scale
 
        p_vals, q_vals, theta_vals = [], [], []
        # Desample images for the current scale
        desampled_img1 = tools.math.desample_image(orig_img1, scale)
        desampled_img2 = tools.math.desample_image(orig_img2, scale)
       
        # Apply Gaussian smoothing if needed
        scaled_gauss_sigma=gauss_sigma*(math.log(scale, 2)+1)
        desampled_img1 = ndimage.gaussian_filter(desampled_img1, sigma=scaled_gauss_sigma)
        desampled_img2 = ndimage.gaussian_filter(desampled_img2, sigma=scaled_gauss_sigma)
   

        print(f"-------- Processing at resolution level: 1/{scale}, gaussian_sigma: {scaled_gauss_sigma} -----------")
        # Updated learning rate
        eta_r = orig_eta_r*(10**math.log(scale, 2))
        eta_t = orig_eta_t*scale
 
        # Optimization loop initialization
        v_p, v_q, v_t = 0, 0, 0  # Initialize velocity terms for momentum
        m_p, m_q, m_t = 0, 0, 0  # Initialize velocity terms for momentum
 
        # Main optimization loop
        for iter_count in range(1, n_iter+1):
            # Store parameter histories for every iteration
            p_hist.append(p)
            q_hist.append(q)
            theta_hist.append(theta)
            scale_hist.append(scale)
           
            # Store parameters and SSD values for analysis
            p_vals.append(p)
            q_vals.append(q)
            theta_vals.append(theta)
           
            # Compute transformation with or without momentum
            new_desampled_img2, coords = rigid_image(desampled_img2, p, q, theta)
            new_img2, _ = rigid_image(orig_img2, p*scale, q*scale, theta)
           
            # Register new image and calculate SSD
            reg_images.append(new_desampled_img2)
            ssd = tools.math.ssd(orig_img1, new_img2)
            ssd_hist.append(ssd)
           
            # Check for stability after new image registration
            if ssd_hist[-1] < conv_thresh or check_stable_ssd(ssd_hist, n_last, max_std):
                break
           
            # Calculate gradients
            x_coords, y_coords = coords[0], coords[1]
            grad_y, grad_x = np.gradient(new_desampled_img2)
            grad_t = np.sum((new_desampled_img2 - desampled_img1) * (-grad_x * y_coords + grad_y * x_coords))
            grad_p = np.sum((new_desampled_img2 - desampled_img1) * grad_x)
            grad_q = np.sum((new_desampled_img2 - desampled_img1) * grad_y)
           
            # Update translation and rotation based on optimizer
            if grad_optimizer == 0:  # Regular gradient descent
                p -= eta_t * grad_p
                q -= eta_t * grad_q
                theta -= eta_r * grad_t
            elif grad_optimizer == 1:  # Momentum-based optimization
                v_p = momentum * v_p + eta_t * grad_p
                v_q = momentum * v_q + eta_t * grad_q
                v_t = momentum * v_t + eta_r * grad_t
               
                p -= v_p
                q -= v_q
                theta -= v_t              
            else:  # Adam optimization
                # Update biased first moment estimate
                m_p = beta_1*m_p + (1-beta_1)*grad_p
                m_q = beta_1*m_q + (1-beta_1)*grad_q
                m_t = beta_1*m_t + (1-beta_1)*grad_t
 
                # Update biased first moment estimate
                v_p = beta_2*v_p + (1-beta_2)*grad_p**2
                v_q = beta_2*v_q + (1-beta_2)*grad_q**2
                v_t = beta_2*v_t + (1-beta_2)*grad_t**2
 
                #Bias correction
                m_hat_p = m_p/(1-beta_1**iter_count)
                m_hat_q = m_q/(1-beta_1**iter_count)
                m_hat_t = m_t/(1-beta_1**iter_count)
               
                v_hat_p = v_p/(1-beta_2**iter_count)
                v_hat_q = v_q/(1-beta_2**iter_count)
                v_hat_t = v_t/(1-beta_2**iter_count)
                
                p -= orig_eta_t*m_hat_p/(np.sqrt(v_hat_p)+epsilon)
                q -= orig_eta_t*m_hat_q/(np.sqrt(v_hat_q)+epsilon)
                theta -= orig_eta_r*m_hat_t/(np.sqrt(v_hat_t)+epsilon)
 
            # Print the current iteration status
            print(
                f"Iter {iter_count:3} | SSD {round(ssd, 2):<10} | "
                f"{'%SSD(aft/bef)':<18}{round(ssd_hist[-1] * 100 / ssd_hist[-2], 3) if len(ssd_hist) >= 2 else 'None':<7} | "
                f"Grad p {round(grad_p, 1):<10} | P {round(p * scale, 2):<6} | "
                f"Grad q {round(grad_q, 1):<10} | Q {round(q * scale, 2):<6} | "
                f"Grad t {round(grad_t, 1):<12} | Theta {round(theta, 3):<6}"
            )
 
        # Store results from this scale and update best parameters
        ssd_hist_all_scales += ssd_hist
        best_idx = np.argmin(np.array(ssd_hist))
        p, q, theta = p_vals[best_idx], q_vals[best_idx], theta_vals[best_idx]
       
    # Return the history of p, q, theta along with other results
    return reg_images, ssd_hist_all_scales, p_hist, q_hist, theta_hist, scale_hist
