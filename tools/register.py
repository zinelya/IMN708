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
from PIL import Image

def is_repeated(percentages, n_last):
    # Check if there are enough elements to verify
    if len(percentages) < n_last:
        return False
    
    # Get the last n_last from percentages
    last_elements = percentages[-n_last:]
    
    # Check for repeating pattern in even and odd indices
    even_values = last_elements[::2]  # Elements at even indices
    odd_values = last_elements[1::2]  # Elements at odd indices
    
    # Verify if all even-indexed elements are the same and all odd-indexed elements are the same
    is_even_repeated = all(x == even_values[0] for x in even_values)
    is_odd_repeated = all(y == odd_values[0] for y in odd_values)
    
    return is_even_repeated and is_odd_repeated and percentages[-1] < percentages[-2]

def check_stable_ssd(history, n_last):
    """
    Checks if the last n_last in the history array are:
    - Increasing
    - Unchanged (within a small tolerance of each other)
    - Decreasing
    Returns True if any of these conditions is met.
    """
    if len(history) < n_last:
        return False  # Not enough elements to check

    if history[-1] > history[-2]:
        return False # Make sure last SSD is less than previous one
    
    last_values = history[-n_last:]
    
    # Calculate differences between consecutive elements
    differences = [last_values[i+1] - last_values[i] for i in range(len(last_values) - 1)]
    
    # Count positive and negative differences
    positives = sum(1 for diff in differences if diff > 0)
    negatives = sum(1 for diff in differences if diff < 0)
    
    # Define threshold for "mostly increasing"
    total_diffs = len(differences)
    # print(positives, negatives, total_diffs)
    if positives > 0.75 * total_diffs:
        return True
    elif negatives > 0.75 * total_diffs:
        return False
    else:
        return True
    
# Translation function
def translate_image(I, p, q):
    # Get the original image dimensions
    h, w = I.shape

    # Create a grid of x and y coordinates
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

    # Apply translation offsets
    x_translated = x + p
    y_translated = y + q

    # Interpolate using map_coordinates
    img_t = ndimage.map_coordinates(I, [y_translated.ravel(), x_translated.ravel()], order=5).reshape(h, w)

    return img_t

def register_translation_ssd(
    image_1, image_2, gradient_optimizer, eta, n_iterations, convergence_value, n_last):
    """
    Register two images by adjusting translation to minimize SSD.
    
    Parameters:
    - image_1: Reference image (numpy array)
    - image_2: Image to be translated (numpy array)
    - gradient_optimizer: Method for gradient descent (update p and q)
    - n_iterations: Maximum number of iterations
    - convergence_value: Threshold for stopping based on SSD
    - n_last: Number of elements to check for unchanged percentages
    - eta: Step size for gradient descent on translation (default: 1e-6)
    
    Returns:
    - registered_images: List of translated images through iterations
    - ssd_history: List of SSD values through iterations
    """
    # Initialize storage for images and SSD values
    registered_images = []
    ssd_history = []
    
    # Initialize translation parameters
    p = 0  # Horizontal translation
    q = 0  # Vertical translation
    
    # Calculate initial SSD and add initial state to lists
    registered_images.append(image_2)
    ssd_history.append(tools.math.ssd(image_1, image_2))
    iteration_count = 0

    # Start optimization loop
    while (
        ssd_history[-1] > convergence_value 
        and iteration_count <= n_iterations 
        and not check_stable_ssd(ssd_history, n_last)
        and not is_repeated(ssd_history, 6)
    ):
        iteration_count += 1
        
        # Apply current translation to the image
        new_J = translate_image(image_2, p, q)
        registered_images.append(new_J)
        
        # Calculate the new SSD value
        new_ssd = tools.math.ssd(image_1, new_J)
        # Append the new SSD value to history and update original SSD
        ssd_history.append(new_ssd)

        # Compute gradients with respect to horizontal and vertical translations
        derive_p = np.sum((new_J - image_1) * np.gradient(new_J, axis=1))  # x-axis gradient
        derive_q = np.sum((new_J - image_1) * np.gradient(new_J, axis=0))  # y-axis gradient

        # Update translations based on gradient optimizer method
        if gradient_optimizer == 0:
            #Update p, q based on sign of gradient
            p -= eta*np.sign(derive_p)
            q -= eta*np.sign(derive_q)
        else:
            #Update p, q based on maginitude of gradient
            p -= eta*derive_p
            q -= eta*derive_q

        # Log current iteration details
        print(
            f"{'Iter':<5}{iteration_count:<3} | "
            f"{'SSD':<5}{round(ssd_history[-1], 2):<10} | "
            f"{'% SSD':<8}{round(ssd_history[-1]*100 / ssd_history[-2], 3) if len(ssd_history) >= 2 else 'None':<6} | "
            f"{'Grad p':<8}{round(derive_p, 3):<10} | "
            f"{'P':<5}{round(p, 2):<8} | "
            f"{'Grad q':<8}{round(derive_q, 3):<10} | "
            f"{'Q':<5}{round(q, 2):<8}"
        )


    return registered_images, ssd_history

# Rotation function
def rotate_image(I, theta):
    # Get the original image dimensions
    h, w = I.shape

    # Create a grid of x and y coordinates
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

    # Apply rotation transformation
    x_rotated = x * np.cos(theta) - y * np.sin(theta)
    y_rotated = x * np.sin(theta) + y * np.cos(theta)

    # Use map_coordinates with rotated coordinates
    img_r = ndimage.map_coordinates(I, [y_rotated.ravel(), x_rotated.ravel()], order=5).reshape(h, w)
    
    return img_r, [x_rotated, y_rotated]

def register_rotation_ssd(
    image_1, image_2, gradient_optimizer, eta_r, n_iterations, convergence_value,
    n_last):
    """
    Register two images by adjusting rotation to minimize SSD.
    
    Parameters:
    - image_1: Reference image (numpy array)
    - image_2: Image to be rotated (numpy array)
    - gradient_optimizer: Method for gradient descent (update theta)
    - eta_r: Learning rate for gradient descent on rotation angle (default: 1e-10)
    - n_iterations: Maximum number of iterations
    - convergence_value: Threshold for stopping based on SSD
    - n_last: Number of elements to check for unchanged percentages
    
    Returns:
    - registered_images: List of rotated images through iterations
    - ssd_history: List of SSD values through iterations
    """
    # Storage for transformed images and SSD values
    registered_images = []
    ssd_history = []
    
    # Initialize rotation angle
    theta = math.pi / 180  # Initial small rotation angle
    
    # Calculate initial SSD and store initial state
    registered_images.append(image_2)
    ssd_history.append(tools.math.ssd(image_1, image_2))
    
    iteration_count = 0

    # Start optimization loop
    while (
        ssd_history[-1] > convergence_value 
        and iteration_count <= n_iterations 
        and not check_stable_ssd(ssd_history, n_last)
        and not is_repeated(ssd_history, 6)
    ):
        iteration_count += 1
        
        # Apply rotation to the image
        new_J, transformed_points_2d = rotate_image(image_2, theta)
        registered_images.append(new_J)
        new_ssd = tools.math.ssd(image_1, new_J) 
        # Update SSD history and check for convergence
        ssd_history.append(new_ssd)

        # Compute gradient with respect to rotation
        x_coordinate = transformed_points_2d[0]
        y_coordinate = transformed_points_2d[1]
        gradient_y, gradient_x = np.gradient(new_J)
        derive_theta = np.sum((new_J - image_1) * (-gradient_x * y_coordinate + gradient_y * x_coordinate))

        # Update rotation angle using gradient descent
        if gradient_optimizer == 0:
            theta -= eta_r*np.sign(derive_theta)
        else:
            theta -= eta_r*derive_theta

        # Print iteration status
        print(
            f"{'Iter':<5}{iteration_count:<3} | "
            f"{'SSD':<5}{round(ssd_history[-1], 2):<10} | "
            f"{'% SSD':<8}{round(ssd_history[-1]*100 / ssd_history[-2], 3) if len(ssd_history) >= 2 else 'None':<6} | "
            f"{'Grad t':<8}{round(derive_theta, 3):<10} | "
            f"{'Theta':<6}{round(theta, 3):<6}"
        )


    return registered_images, ssd_history

# Ridgid function
def rigid_image(I, p, q, theta):
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

    return img_r, [x_rotated, y_rotated]

def register_rigid_ssd(
    orig_img1, orig_img2, optimizer, eta_t, eta_r, max_iter, conv_thresh, 
    scale_max, gauss_sigma, n_last
):
    # Initialize transformation parameters
    reg_images = []  # Store registered images for each scale
    ssd_hist_all_scales = []
    p, q, theta = 0, 0, 0  # Initial translations and rotation
    momentum = 0.9  # Momentum factor for gradient updates

    # Set scaling factors for multiscale processing
    scales = [2**s for s in range(scale_max + 1)][::-1]

    for s, scale in enumerate(scales):
        print(f"-------- Processing at Scale: {scale} -----------")
        
        ssd_hist = []  # Store SSD values at this scale
        p_vals, q_vals, theta_vals = [], [], []

        # Adjust translations and rotation for this scale
        if s > 0:
            p = scales[s - 1] * p / scale
            q = scales[s - 1] * q / scale

        # Resize images for the current scale
        img1 = Image.fromarray(orig_img1.astype(np.int8))  # Convert to uint8 for PIL
        img2 = Image.fromarray(orig_img2.astype(np.int8))
        h1, w1 = orig_img1.shape
        h2, w2 = orig_img2.shape
        img1_resized = img1.resize((w1 // scale, h1 // scale), Image.LANCZOS)
        img2_resized = img2.resize((w2 // scale, h2 // scale), Image.LANCZOS)
        img1_arr = np.array(img1_resized).astype(int)
        img2_arr = np.array(img2_resized).astype(int)
        
        # Apply Gaussian smoothing if needed
        if gauss_sigma and scale > 1:
            img1_arr = ndimage.gaussian_filter(img1_arr, sigma=gauss_sigma)
            img2_arr = ndimage.gaussian_filter(img2_arr, sigma=gauss_sigma)

        # Adjust learning rates for smaller scales
        if optimizer >= 2 and scale == 1:
            eta_t, eta_r = eta_t * 0.01, eta_r * 0.01
        elif optimizer >= 2 and scale == 2:
            eta_t, eta_r = eta_t * 0.1, eta_r * 0.1
            
        # Optimization loop initialization
        iter_count = 0
        v_p_old, v_q_old, v_theta_old = 0, 0, 0  # Initialize velocity terms for momentum

        # Initial SSD calculation
        ssd = tools.math.ssd(orig_img1, orig_img2)
        img_h, img_w = img1_arr.shape

        # Main optimization loop
        while (
            ssd > conv_thresh 
            and iter_count <= max_iter 
            and not check_stable_ssd(ssd_hist, n_last)
            and not is_repeated(ssd_hist, 6)
        ):
            iter_count += 1
            
            # Compute transformation with or without momentum
            if optimizer != 3:
                new_img2, coords = rigid_image(img2_arr, p, q, theta)
                non_resized_new_img2, _ = rigid_image(orig_img2, p * scale, q * scale, theta)
            else:
                new_img2, coords = rigid_image(img2_arr, p - momentum * v_p_old, q - momentum * v_q_old, theta - momentum * v_theta_old)
                non_resized_new_img2, _ = rigid_image(orig_img2, (p - momentum * v_p_old) * scale, (q - momentum * v_q_old) * scale, theta - momentum * v_theta_old)

            # Register new image and calculate SSD
            reg_images.append(new_img2)
            ssd = tools.math.ssd(orig_img1, non_resized_new_img2)
            ssd_hist.append(ssd)
            # ssd = tools.math.ssd(img1_arr, new_img2)
            x_coords, y_coords = coords[0], coords[1]

            # Calculate gradients
            grad_y, grad_x = np.gradient(new_img2)
            grad_theta = np.sum((new_img2 - img1_arr) * (-grad_x * y_coords + grad_y * x_coords))
            grad_p = np.sum((new_img2 - img1_arr) * grad_x)
            grad_q = np.sum((new_img2 - img1_arr) * grad_y)
            
            # Update translation and rotation based on optimizer
            if optimizer == 0:  # Fixed step
                p -= eta_t * np.sign(grad_p)
                q -= eta_t * np.sign(grad_q)
                theta -= eta_r * np.sign(grad_theta)
            elif optimizer == 1:  # Regular gradient descent
                p -= eta_t * grad_p
                q -= eta_t * grad_q
                theta -= eta_r * grad_theta
            elif optimizer in (2, 3):  # Momentum-based optimization
                v_p_new = momentum * v_p_old + eta_t * grad_p
                v_q_new = momentum * v_q_old + eta_t * grad_q
                v_theta_new = momentum * v_theta_old + eta_r * grad_theta
                
                p -= v_p_new
                q -= v_q_new
                theta -= v_theta_new
                
                # Update velocity values for momentum
                v_p_old, v_q_old, v_theta_old = v_p_new, v_q_new, v_theta_new
            
            # Store parameters and SSD values for analysis
            p_vals.append(p)
            q_vals.append(q)
            theta_vals.append(theta)

            # Print the current iteration status
            print(
                f"Iter {iter_count:3} | SSD {round(ssd, 2):<10} | "
                f"% SSD {round(ssd_hist[-1]*100/ ssd_hist[-2], 3) if len(ssd_hist) >= 2 else 'None':<4} | "
                f"Grad p {round(grad_p, 1):<10} | P {round(p * scale, 2):<6} | "
                f"Grad q {round(grad_q, 1):<10} | Q {round(q * scale, 2):<6} | "
                f"Grad theta {round(grad_theta, 1):<12} | Theta {round(theta, 3):<6}"
            )

        # Store results from this scale and update best parameters
        ssd_hist_all_scales += ssd_hist
        best_idx = np.argmin(np.array(ssd_hist))
        p, q, theta = p_vals[best_idx], q_vals[best_idx], theta_vals[best_idx]
        
    return reg_images, ssd_hist_all_scales, p, q, theta