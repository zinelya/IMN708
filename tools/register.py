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

# Rotation function
def ridgid_image(I, p, q, theta):
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

def is_nearly_repeated(percentages, n_elements):
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

def has_unchanged_percentages(history, n_elements):
    # Check if the last n_elements values are increasing
    if len(history) >= n_elements:
        last_values = history[-n_elements:] 
        # Check if each element is greater than the previous one
        is_decreasing = all(last_values[i] > last_values[i+1] for i in range(0, len(last_values)-1))
        if is_decreasing:
            print('Decreasing')
            return False
    
        is_increasing = all(last_values[i] <= last_values[i+1] for i in range(0, len(last_values)-1))
        if is_increasing:
            print('Increasing')
            return True
    
        is_unchanged = all(0.995 <= last_values[i]/last_values[i+1] <= 1.005 for i in range(0, len(last_values)-1))
        if is_unchanged:
            print('Unchanged')
            return True
        
        return False
        
    else:
        return False  # Not enough elements to check the condition

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

def register_translation_ssd(I, J, gradient_optimizer, n_iterations, convergence_value, bins):
    registered_images = []
    ssd_arr = []
    
    # Initialize translation parameters
    p = 100
    q = 80
    is_stop = False
    count = 0
    eta = 10**(-6)
    gamma = 0.1
    v_p_old, v_q_old = 0, 0
    
    registered_images.append(J)
    ssd_arr.append(tools.math.ssd(I, J))
    
    while ssd_arr[-1] > convergence_value and count <= n_iterations:
        count += 1
        
        if gradient_optimizer != 3:
            new_J = translate_image(J, p, q)
        else:
            new_J, transformed_points_2d = translate_image(J, p-gamma*v_p_old, q-gamma*v_q_old)
         
        registered_images.append(new_J)
        new_ssd = tools.math.ssd(I, new_J)
        
        derive_p = np.sum((new_J - I) * np.gradient(new_J, axis=1))
        derive_q = np.sum((new_J - I) * np.gradient(new_J, axis=0))
        
        if gradient_optimizer == 0:
            p -= np.sign(derive_p)
            q -= np.sign(derive_q)
        
        elif gradient_optimizer == 1:
            p -= eta*derive_p
            q -= eta*derive_q
            
        elif gradient_optimizer == 2 or gradient_optimizer == 3:
            v_p_new = gamma*v_p_old + eta*derive_p
            v_q_new = gamma*v_q_old + eta*derive_q
            
            p -= v_p_new
            q -= v_q_new
            
            v_p_old = v_p_new
            v_q_old = v_q_new
                      
        print(f'Loop {count-1}:', new_ssd, p, q)  
        ssd_arr.append(new_ssd)

    return registered_images, ssd_arr  # Correct return statement

def register_rotation_ssd(I, J, gradient_optimizer, n_iterations, convergence_value, bins):
    registered_images = []
    ssd_arr = []
    
    # Initialize translation parameters
    theta = math.pi/180
    is_stop = False
    count = 0
    eta = 10**(-10)
    gamma = 0.1
    v_t_old = 0
    
    registered_images.append(J)
    ssd_arr.append(tools.math.ssd(I, J))
    h, w = I.shape
    
    while ssd_arr[-1] > convergence_value and count <= n_iterations:
        count += 1
        
        if gradient_optimizer != 3:
            new_J, transformed_points_2d = rotate_image(J, theta)
        else:
            new_J, transformed_points_2d = rotate_image(J, theta-gamma*v_t_old)

        registered_images.append(new_J)
        new_ssd = tools.math.ssd(I, new_J)
        x_coordinate = transformed_points_2d[0]
        y_coordinate = transformed_points_2d[1]

        gradient_y, gradient_x = np.gradient(new_J)
        derive_theta = np.sum((new_J-I)*(-gradient_x*y_coordinate+gradient_y*x_coordinate))

        if derive_theta == 0:
            is_stop = True
        else:
            if gradient_optimizer == 0:
                theta -= np.sign(derive_theta)*math.pi/180
            
            elif gradient_optimizer == 1:
                theta -= eta*derive_theta
                
            elif gradient_optimizer == 2 or gradient_optimizer == 3:
                v_t_new = gamma*v_t_old + eta*derive_theta
                theta -= v_t_new
                v_t_old = v_t_new
                        
        print(f'Loop {count-1}:', new_ssd, theta)
        ssd_arr.append(new_ssd)

    return registered_images, ssd_arr  # Correct return statement

def register_rigid_ssd(image_1, image_2, gradient_optimizer, n_iterations, convergence_value, resize_factor, gaussian_sigma, n_elements, bins):
    non_resize_I = np.array(image_1).astype(int)
    non_resize_J = np.array(image_2).astype(int)

    # Initialize translation parameters
    registered_images = []
    ssd_arrs = []
    p, q = 0, 0
    theta = 0
    eta_1 = 10**(-6)
    eta_2 = 10**(-9)
    gamma = 0.9
    
    if resize_factor > 1:
        scale_factors = [2**scale_factor for scale_factor in range(0, resize_factor+1)][::-1]
    else:
        scale_factors = [1]
        
    for i, scale_factor in enumerate(scale_factors):
        print(f'-------- Resize factor: {scale_factor} ---------------')
        ssd_arr = []
        ps = []
        qs = []
        ts = []
        
        p = scale_factors[i-1]*p/scale_factor
        q = scale_factors[i-1]*q/scale_factor
        w_1, h_1 = image_1.size
        w_2, h_2 = image_2.size
        image_1_resized = image_1.resize((w_1 // scale_factor, h_1 // scale_factor), Image.LANCZOS)
        image_2_resized = image_2.resize((w_2 // scale_factor, h_2 // scale_factor), Image.LANCZOS)
        I = np.array(image_1_resized).astype(int)
        J = np.array(image_2_resized).astype(int)
        
        if gaussian_sigma and scale_factor > 1:
            I = ndimage.gaussian_filter(I, sigma=gaussian_sigma)
            J = ndimage.gaussian_filter(J, sigma=gaussian_sigma)
            
        count = 0
        v_p_old, v_q_old, v_t_old = 0, 0, 0
        
        non_resize_ssd = tools.math.ssd(non_resize_I, non_resize_J)
        h, w = I.shape

        while non_resize_ssd > convergence_value and count <= n_iterations and not has_unchanged_percentages(ssd_arr, n_elements) and not is_nearly_repeated(ssd_arr, 6):
            count += 1
            
            if gradient_optimizer != 3:
                new_J, transformed_points_2d = ridgid_image(J, p, q, theta)
                non_resize_new_J, _ = ridgid_image(non_resize_J, p*scale_factor, q*scale_factor, theta)
            else:
                new_J, transformed_points_2d = ridgid_image(J, p-gamma*v_p_old, q-gamma*v_q_old, theta-gamma*v_t_old)
                non_resize_new_J, _ = ridgid_image(non_resize_J, (p-gamma*v_p_old)*scale_factor, (q-gamma*v_p_old)*scale_factor, theta-gamma*v_t_old)

            registered_images.append(non_resize_new_J)
            new_ssd = tools.math.ssd(I, new_J)
            x_coordinate = transformed_points_2d[0]
            y_coordinate = transformed_points_2d[1]

            gradient_y, gradient_x = np.gradient(new_J)
            derive_theta = np.sum((new_J - I)*(-gradient_x*y_coordinate+gradient_y*x_coordinate))
            derive_p = np.sum((new_J - I) * gradient_x)
            derive_q = np.sum((new_J - I) * gradient_y)
            
            if gradient_optimizer == 0 or scale_factor == 1:
                p -= np.sign(derive_p)
                q -= np.sign(derive_q)
                theta -= np.sign(derive_theta)*math.pi/1800
            elif gradient_optimizer == 1:
                p -= eta_1*derive_p
                q -= eta_1*derive_q
                theta -= eta_2*derive_theta
            elif gradient_optimizer == 2 or gradient_optimizer == 3:
                v_p_new = gamma*v_p_old + eta_1*derive_p
                v_q_new = gamma*v_q_old + eta_1*derive_q
                v_t_new = gamma*v_t_old + eta_2*derive_theta
                
                p -= v_p_new
                q -= v_q_new
                theta -= v_t_new
                
                v_p_old = v_p_new
                v_q_old = v_q_new
                v_t_old = v_t_new

            ps.append(p)
            qs.append(q)
            ts.append(theta)
            
            non_resize_ssd = tools.math.ssd(non_resize_I, non_resize_new_J)
            ssd_arr.append(non_resize_ssd)
            print(
                f"{'Loop':<5}{count-1:<5} |"
                f"{'SSD':<5}{round(non_resize_ssd, 2):<11} |"
                f"{'%SSD':<7}{round(ssd_arr[-2] / ssd_arr[-1], 3) if len(ssd_arr) >= 2 else 'None':<6} |"
                f"{'Derive p':<10}{round(derive_p, 3):<10} |"
                f"{'P':<5}{round(p * scale_factor, 2):<6} |"
                f"{'Derive q':<10}{round(derive_q, 3):<10} |"
                f"{'Q':<5}{round(q * scale_factor, 2):<6} |"
                f"{'Derive t':<10}{round(derive_theta, 3):<14} |"
                f"{'T ':<5}{round(theta, 3):<6}"
            )
            
        ssd_arrs += ssd_arr
        min_pos = np.argmin(np.array(ssd_arr))
        p = ps[min_pos]
        q = qs[min_pos]
        theta = ts[min_pos]
        
    return registered_images, ssd_arrs, p, q, theta  # Correct return statement
    