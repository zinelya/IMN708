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
    p_translation = 0  # Horizontal translation
    q_translation = 0  # Vertical translation
    
    # Calculate initial SSD and add initial state to lists
    registered_images.append(image_2)
    ssd_history.append(tools.math.ssd(image_1, image_2))
    cur_ssd = ssd_history[-1]
    iteration_count = 0

    # Start optimization loop
    while (
        cur_ssd > convergence_value 
        and iteration_count <= n_iterations 
        and not check_stable_ssd(ssd_history, n_last)
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

        # Update translations based on gradient optimizer method
        if gradient_optimizer == 0:
            #Update p, q based on sign of gradient
            p_translation -= eta*np.sign(derive_p)
            q_translation -= eta*np.sign(derive_q)
        else:
            #Update p, q based on maginitude of gradient
            p_translation -= eta*derive_p
            q_translation -= eta*derive_q

        # Log current iteration details
        print(
            f"{'Iter':<5}{iteration_count:<3} | "
            f"{'SSD':<5}{round(cur_ssd, 2):<10} | "
            f"{'% SSD':<8}{round(ssd_history[-2] / ssd_history[-1], 3) if len(ssd_history) >= 2 else 'None':<6} | "
            f"{'Grad p':<8}{round(derive_p, 3):<10} | "
            f"{'P':<5}{round(p_translation, 2):<8} | "
            f"{'Grad q':<8}{round(derive_q, 3):<10} | "
            f"{'Q':<5}{round(q_translation, 2):<8}"
        )

        # Append the new SSD value to history and update original SSD
        ssd_history.append(new_ssd)
        cur_ssd = new_ssd

    return registered_images, ssd_history