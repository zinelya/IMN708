import matplotlib.pyplot as plt
import numpy as np
import scipy
import tools.math

def translate_image(I, p, q):
    m, n = I.shape
    
    x_coords, y_coords = np.meshgrid(np.arange(n), np.arange(m))
    points_2d = np.column_stack((x_coords.ravel(), y_coords.ravel(), I.ravel()))
    flat_intensity = I.ravel()
    
    # Translate points:
    translated_points_2d = np.copy(points_2d).astype(float)
    translated_points_2d[:, 0] += p  # Add p to all x-coordinates
    translated_points_2d[:, 1] += q  # Add q to all y-coordinates
    
    # Interpolation using vectorization
    new_image = np.zeros_like(I)  # Initialize new image
    min_intensity = np.min(flat_intensity)
    
    neighbor_counts = []
    
    for idx_1, point in enumerate(points_2d):
        x_1, y_1 = point[0], point[1]
        
        # Get translated points within the delta range
        x_diff = translated_points_2d[:, 0] - x_1
        y_diff = translated_points_2d[:, 1] - y_1
        mask = (np.abs(x_diff) < 1) & (np.abs(y_diff) < 1)

        # Calculate new value for I[x_1, y_1]
        if np.any(mask):  # Only proceed if there are valid points
            translated_intensities = flat_intensity[mask]
            delta_x = x_diff[mask]
            delta_y = y_diff[mask]
            neighbor_counts.append(delta_x.shape)
            new_value = np.sum((1-delta_x) * (1-delta_y) * translated_intensities)
            new_image[y_1, x_1] = new_value
        else:
            new_image[y_1, x_1] = min_intensity
            
    print(np.unique(np.array(neighbor_counts)))
    return new_image

def rotate_image(I, theta):
    m, n = I.shape
    
    x_coords, y_coords = np.meshgrid(np.arange(n), np.arange(m))
    points_2d = np.column_stack((x_coords.ravel(), y_coords.ravel(), I.ravel()))
    flat_intensity = I.ravel()
    
    # Convert theta from degrees to radians
    theta_rad = np.radians(theta)
    # Rotation matrix
    rotation_matrix = np.array([[np.cos(theta_rad), -np.sin(theta_rad)],
                                 [np.sin(theta_rad), np.cos(theta_rad)]])
    
    # Interpolation using vectorization
    rotated_points_2d = points_2d[:, :2] @ rotation_matrix.T  # Apply rotation
    new_image = np.zeros_like(I)  # Initialize new image
    min_intensity = np.min(flat_intensity)
    
    neighbor_counts = []
    
    for idx_1, point in enumerate(points_2d):
        x_1, y_1 = point[0], point[1]
        # Get rotated points within the delta range
        x_diff = rotated_points_2d[:, 0] - x_1
        y_diff = rotated_points_2d[:, 1] - y_1
        mask = (np.abs(x_diff) < 1) & (np.abs(y_diff) < 1)

        # Calculate new value for I[x_1, y_1]
        if np.any(mask):  # Only proceed if there are valid points
            rotated_intensities = flat_intensity[mask]
            delta_x = x_diff[mask]
            delta_y = y_diff[mask]
            neighbor_counts.append(delta_x.shape)
            new_value = np.sum((1-delta_x) * (1-delta_y) * rotated_intensities)
            new_image[y_1, x_1] = new_value
        else:
            new_image[y_1, x_1] = min_intensity
            
    print(np.unique(np.array(neighbor_counts)))
    return new_image

def register_translation_ssd(I, J, bins):
    registered_images = []
    ssd_arr = []
    
    loop_count = 1
    while loop_count <= 3:
        J = translate_image(J, 7.5, 7.5)
        registered_images.append(J)
        ssd_arr.append(tools.math.ssd(I, J))
        loop_count += 1
        
    return registered_images, ssd_arr

def register_rotation_ssd(I, J, bins):
    registered_images = []
    ssd_arr = []
    
    loop_count = 1
    
    registered_images.append(J)
    ssd_arr.append(tools.math.ssd(I, J))
        
    while loop_count <= 5:
        J = rotate_image(J, 1)
        registered_images.append(J)
        ssd_arr.append(tools.math.ssd(I, J))
        loop_count += 1
        
    return registered_images, ssd_arr

def register_rigid_ssd(I, J, bins):
    return [I, I], [1, 1]