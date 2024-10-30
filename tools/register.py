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

def is_repeated(percentages, n_elements):
    # Check if there are enough elements to verify
    if len(percentages) < n_elements:
        return False
    
    # Get the last n_elements from percentages
    last_elements = percentages[-n_elements:]
    
    # Check for repeating pattern in even and odd indices
    even_values = last_elements[::2]  # Elements at even indices
    odd_values = last_elements[1::2]  # Elements at odd indices
    
    # Verify if all even-indexed elements are the same and all odd-indexed elements are the same
    is_even_repeated = all(x == even_values[0] for x in even_values)
    is_odd_repeated = all(y == odd_values[0] for y in odd_values)
    
    return is_even_repeated and is_odd_repeated and percentages[-1] < percentages[-2]

def check_stable_ssd(history, n_elements):
    """
    Checks if the last n_elements in the history array are:
    - Increasing
    - Unchanged (within a small tolerance of each other)
    - Decreasing
    Returns True if any of these conditions is met.
    """
    if len(history) < n_elements:
        return False  # Not enough elements to check

    last_values = history[-n_elements:]
    
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
    
    
    # # Check if the sequence is strictly decreasing
    # is_decreasing = all(last_values[i] > last_values[i + 1] for i in range(len(last_values) - 1))
    # if is_decreasing:
    #     return False
    
    # # Check if the sequence is strictly increasing
    # is_increasing = all(last_values[i] <= last_values[i + 1] for i in range(len(last_values) - 1))
    # if is_increasing:
    #     print("Increasing")
    #     return True
    
    # # Check if the sequence is nearly unchanged
    # tolerance = 0.005
    # is_unchanged = all(1 - tolerance <= last_values[i] / last_values[i + 1] <= 1 + tolerance for i in range(len(last_values) - 1))
    # if is_unchanged:
    #     return True
    
    # return False

def register_translation_ssd(
    image_1, image_2, n_iterations, convergence_value, n_elements):
    """
    Register two images by adjusting translation to minimize SSD.
    
    Parameters:
    - image_1: Reference image (numpy array)
    - image_2: Image to be translated (numpy array)
    - n_iterations: Maximum number of iterations
    - convergence_value: Threshold for stopping based on SSD
    - n_elements: Number of elements to check for unchanged percentages
    - eta: Step size for gradient descent on translation (default: 1e-6)
    
    Returns:
    - registered_images: List of translated images through iterations
    - ssd_history: List of SSD values through iterations
    """
    # Initialize storage for images and SSD values
    registered_images = []
    ssd_history = []
    
    # Initialize translation parameters
    p_translation = 0  # Horizontal translation
    q_translation = 0  # Vertical translation
    
    # Calculate initial SSD and add initial state to lists
    registered_images.append(image_2)
    ssd_history.append(tools.math.ssd(image_1, image_2))
    original_ssd = ssd_history[-1]
    iteration_count = 0

    # Start optimization loop
    while (
        original_ssd > convergence_value 
        and iteration_count <= n_iterations 
        and not check_stable_ssd(ssd_history, n_elements)
        and not is_repeated(ssd_history, 6)
    ):
        iteration_count += 1
        
        # Apply current translation to the image
        new_J = translate_image(image_2, p_translation, q_translation)
        registered_images.append(new_J)
        
        # Calculate the new SSD value
        new_ssd = tools.math.ssd(image_1, new_J)

        # Compute gradients with respect to horizontal and vertical translations
        derive_p = np.sum((new_J - image_1) * np.gradient(new_J, axis=1))  # x-axis gradient
        derive_q = np.sum((new_J - image_1) * np.gradient(new_J, axis=0))  # y-axis gradient

        # Update translations based on gradients
        p_translation -= np.sign(derive_p)
        q_translation -= np.sign(derive_q)

        # Log current iteration details
        print(
            f"{'Iter':<5}{iteration_count:<3} | "
            f"{'SSD':<5}{round(original_ssd, 2):<10} | "
            f"{'% SSD':<8}{round(ssd_history[-2] / ssd_history[-1], 3) if len(ssd_history) >= 2 else 'None':<6} | "
            f"{'Grad p':<8}{round(derive_p, 3):<10} | "
            f"{'P':<5}{round(p_translation, 2):<8} | "
            f"{'Grad q':<8}{round(derive_q, 3):<10} | "
            f"{'Q':<5}{round(q_translation, 2):<8}"
        )

        # Append the new SSD value to history and update original SSD
        ssd_history.append(new_ssd)
        original_ssd = new_ssd

    return registered_images, ssd_history

def register_rotation_ssd(
    image_1, image_2, n_iterations, convergence_value,
    n_elements):
    """
    Register two images by adjusting rotation to minimize SSD.
    
    Parameters:
    - image_1: Reference image (numpy array)
    - image_2: Image to be rotated (numpy array)
    - n_iterations: Maximum number of iterations
    - convergence_value: Threshold for stopping based on SSD
    - n_elements: Number of elements to check for unchanged percentages
    - eta: Learning rate for gradient descent on rotation angle (default: 1e-10)
    
    Returns:
    - registered_images: List of rotated images through iterations
    - ssd_history: List of SSD values through iterations
    """
    # Storage for transformed images and SSD values
    registered_images = []
    ssd_history = []
    
    # Initialize rotation angle
    rotation_theta = math.pi / 180  # Initial small rotation angle
    
    # Calculate initial SSD and store initial state
    registered_images.append(image_2)
    ssd_history.append(tools.math.ssd(image_1, image_2))
    
    iteration_count = 0
    original_ssd = ssd_history[-1]

    # Start optimization loop
    while (
        original_ssd > convergence_value 
        and iteration_count <= n_iterations 
        and not check_stable_ssd(ssd_history, n_elements)
        and not is_repeated(ssd_history, 6)
    ):
        iteration_count += 1
        
        # Apply rotation to the image
        new_J, transformed_points_2d = rotate_image(image_2, rotation_theta)
        
        registered_images.append(new_J)
        new_ssd = tools.math.ssd(image_1, new_J)

        # Compute gradient with respect to rotation
        x_coordinate = transformed_points_2d[0]
        y_coordinate = transformed_points_2d[1]
        gradient_y, gradient_x = np.gradient(new_J)
        derive_theta = np.sum((new_J - image_1) * (-gradient_x * y_coordinate + gradient_y * x_coordinate))

        # Update rotation angle using gradient descent
        rotation_theta -= np.sign(derive_theta)*math.pi / 360

        # Print iteration status
        print(
            f"{'Iter':<5}{iteration_count:<3} | "
            f"{'SSD':<5}{round(original_ssd, 2):<10} | "
            f"{'% SSD':<8}{round(ssd_history[-2] / ssd_history[-1], 3) if len(ssd_history) >= 2 else 'None':<6} | "
            f"{'Grad t':<8}{round(derive_theta, 3):<10} | "
            f"{'Theta':<6}{round(rotation_theta, 3):<6}"
        )

        # Update SSD history and check for convergence
        ssd_history.append(new_ssd)
        original_ssd = new_ssd

    return registered_images, ssd_history

def register_rigid_ssd(
    image_1, image_2, gradient_optimizer, n_iterations, convergence_value, 
    resize_factor, gaussian_sigma, n_elements, eta_1=1e-6, eta_2=1e-9
):
    # Convert images to integer arrays for processing
    original_img1 = np.array(image_1).astype(int)
    original_img2 = np.array(image_2).astype(int)

    # Initialize parameters
    registered_images = []  # List to store each registered image at different scales
    ssd_history_all_scales = []
    p_translation, q_translation, rotation_theta = 0, 0, 0
    momentum = 0.9  # Learning rates and momentum
    # Set scaling factors for multiscale processing
    scale_factors = [2**scale for scale in range(0, resize_factor + 1)][::-1]

    for scale_index, scale_factor in enumerate(scale_factors):
        print(f"-------- Processing at Scale Factor: {scale_factor} -----------")
        
        ssd_history = []  # Store SSD values for this scale
        p_list, q_list, theta_list = [], [], []
        
        # Adjust translations and rotation for this scale
        p_translation = scale_factors[scale_index - 1] * p_translation / scale_factor if scale_index > 0 else p_translation
        q_translation = scale_factors[scale_index - 1] * q_translation / scale_factor if scale_index > 0 else q_translation

        # Resize images for the current scale
        width_1, height_1 = image_1.size
        width_2, height_2 = image_2.size
        img1_resized = image_1.resize((width_1 // scale_factor, height_1 // scale_factor), Image.LANCZOS)
        img2_resized = image_2.resize((width_2 // scale_factor, height_2 // scale_factor), Image.LANCZOS)
        img1_array = np.array(img1_resized).astype(int)
        img2_array = np.array(img2_resized).astype(int)
        
        # Apply Gaussian smoothing if required
        if gaussian_sigma and scale_factor > 1:
            img1_array = ndimage.gaussian_filter(img1_array, sigma=gaussian_sigma)
            img2_array = ndimage.gaussian_filter(img2_array, sigma=gaussian_sigma)

        if scale_factor == 1:
            eta_1 = eta_1*(10**(-2))
            eta_2 = eta_2*(10**(-2))
        elif scale_factor == 2:
            eta_1 = eta_1*(10**(-1))
            eta_2 = eta_2*(10**(-1))
            
        # Optimization loop initialization
        iteration_count = 0
        v_p_old, v_q_old, v_theta_old = 0, 0, 0  # Initialize velocity terms for momentum

        # Initial SSD calculation
        original_ssd = tools.math.ssd(original_img1, original_img2)
        img_height, img_width = img1_array.shape

        # Main optimization loop
        while (
            original_ssd > convergence_value 
            and iteration_count <= n_iterations 
            and not check_stable_ssd(ssd_history, n_elements)
            and not is_repeated(ssd_history, 6)
        ):
            iteration_count += 1
            
            # Compute transformation with or without momentum
            if gradient_optimizer != 3:
                new_img2, transformed_coords = rigid_image(img2_array, p_translation, q_translation, rotation_theta)
                non_resized_new_img2, _ = rigid_image(original_img2, p_translation * scale_factor, q_translation * scale_factor, rotation_theta)
            else:
                new_img2, transformed_coords = rigid_image(img2_array, p_translation - momentum * v_p_old, q_translation - momentum * v_q_old, rotation_theta - momentum * v_theta_old)
                non_resized_new_img2, _ = rigid_image(original_img2, (p_translation - momentum * v_p_old) * scale_factor, (q_translation - momentum * v_q_old) * scale_factor, rotation_theta - momentum * v_theta_old)

            # Register new image and calculate SSD
            registered_images.append(new_img2)
            # registered_images.append(non_resized_new_img2)
            new_ssd = tools.math.ssd(img1_array, new_img2)
            x_coords, y_coords = transformed_coords[0], transformed_coords[1]

            # Calculate gradients
            gradient_y, gradient_x = np.gradient(new_img2)
            derive_theta = np.sum((new_img2 - img1_array) * (-gradient_x * y_coords + gradient_y * x_coords))
            derive_p = np.sum((new_img2 - img1_array) * gradient_x)
            derive_q = np.sum((new_img2 - img1_array) * gradient_y)
            
            # Update translation and rotation based on optimizer
            if gradient_optimizer == 0: # Fix step
                p_translation -= np.sign(derive_p)
                q_translation -= np.sign(derive_q)
                rotation_theta -= np.sign(derive_theta) * math.pi / 1800
            elif gradient_optimizer == 1: # Regular gradient descent
                p_translation -= eta_1 * derive_p
                q_translation -= eta_1 * derive_q
                rotation_theta -= eta_2 * derive_theta
            elif gradient_optimizer == 2 or gradient_optimizer == 3: # Momentum
                v_p_new = momentum * v_p_old + eta_1 * derive_p
                v_q_new = momentum * v_q_old + eta_1 * derive_q
                v_theta_new = momentum * v_theta_old + eta_2 * derive_theta
                
                p_translation -= v_p_new
                q_translation -= v_q_new
                rotation_theta -= v_theta_new
                
                # Update old velocity values
                v_p_old, v_q_old, v_theta_old = v_p_new, v_q_new, v_theta_new
                
            
            # Store parameters and SSD values for analysis
            p_list.append(p_translation)
            q_list.append(q_translation)
            theta_list.append(rotation_theta)
            original_ssd = tools.math.ssd(original_img1, non_resized_new_img2)
            ssd_history.append(original_ssd)

            # Print the current iteration status
            print(
                f"{'Iter':<5}{iteration_count:<3} | "
                f"{'SSD':<5}{round(original_ssd, 2):<10} | "
                f"{'% SSD':<7}{round(ssd_history[-2] / ssd_history[-1], 3) if len(ssd_history) >= 2 else 'None':<4} | "
                f"{'Grad p':<8}{round(derive_p, 1):<10} | "
                f"{'P':<3}{round(p_translation * scale_factor, 2):<6} | "
                f"{'Grad q':<8}{round(derive_q, 1):<10} | "
                f"{'Q':<3}{round(q_translation * scale_factor, 2):<6} | "
                f"{'Grad t':<8}{round(derive_theta, 1):<12} | "
                f"{'Theta':<6}{round(rotation_theta, 3):<6}"
            )

        # Store results from this scale and update best parameters
        ssd_history_all_scales += ssd_history
        best_index = np.argmin(np.array(ssd_history))
        p_translation, q_translation, rotation_theta = p_list[best_index], q_list[best_index], theta_list[best_index]
        
    return registered_images, ssd_history_all_scales, p_translation, q_translation, rotation_theta