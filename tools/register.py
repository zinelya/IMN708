import matplotlib.pyplot as plt
import numpy as np
import scipy
import tools.math
import tools.display
import matplotlib.patches as patches
from scipy import interpolate
from scipy import ndimage
import math

# Translation function
def translate_image(I, p, q):
    x = np.arange(0, I.shape[0]) + p
    y = np.arange(0, I.shape[1]) + q
    xv, yv = np.meshgrid(x, y)
    coord = [xv, yv]
    img_t = ndimage.map_coordinates(I, coord, order=5).T

    return img_t

# Rotation function
def rotate_image(I, theta):
    theta = -theta
    T = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])

    x = np.arange(0, I.shape[0])
    y = np.arange(0, I.shape[1])
    xv, yv = np.meshgrid(x, y)
    # grid with rotation
    xv_t = T[0, 0] * xv + T[0, 1] * yv
    yv_t = T[1, 0] * xv + T[1, 1] * yv
    coord = [xv_t, yv_t]

    img_r = ndimage.map_coordinates(I, coord, order=5).T

    return img_r, np.stack((xv_t.ravel(), yv_t.ravel()), axis=-1)

# Rotation function
def ridgid_image(I, p, q, theta):
    theta = -theta
    T = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    # original grid
    x = np.arange(0, I.shape[0])
    y = np.arange(0, I.shape[1])
    xv, yv = np.meshgrid(x, y)
    # grid with rotation
    xv_t = T[0, 0] * xv + T[0, 1] * yv + p
    yv_t = T[1, 0] * xv + T[1, 1] * yv + q
    coord = [xv_t, yv_t]

    img_r = ndimage.map_coordinates(I, coord, order=5).T

    return img_r, np.stack((xv_t.ravel(), yv_t.ravel()), axis=-1)

def has_converged_or_plateaued(history, new_ssd):
    if new_ssd < 30000000:
        return True
    # if new_ssd <= history[-1]:
    #     min_ssd = np.min(np.array(history))
    #     repeated_ssd = [ssd for ssd in history if 0.9 <= ssd/new_ssd <= 1.1]
    #     if len(repeated_ssd) >= 500:
    #         return True

    return False

def register_translation_ssd(I, J, gradient_optimizer, bins):
    registered_images = []
    ssd_arr = []
    
    # Initialize translation parameters
    p = -5
    q = -5
    is_stop = False
    count = 0
    eta = 10**(-6)
    gamma = 0.9
    v_p_old, v_q_old = 0, 0
    
    registered_images.append(J)
    ssd_arr.append(tools.math.ssd(I, J))
    
    while not is_stop and count <= 100:
        count += 1
        
        if gradient_optimizer != 3:
            new_J = translate_image(J, p, q)
        else:
            new_J, transformed_points_2d = translate_image(J, p-gamma*v_p_old, q-gamma*v_q_old)
            
        registered_images.append(new_J)
        new_ssd = tools.math.ssd(I, new_J)
        
        if has_converged_or_plateaued(ssd_arr, new_ssd):
            is_stop = True
        else:
            derive_p = np.sum((new_J - I) * np.gradient(new_J, axis=0))
            derive_q = np.sum((new_J - I) * np.gradient(new_J, axis=1))
            
            if gradient_optimizer == 0:
                p -= np.sign(derive_p)
                q -= np.sign(derive_q)
            
            elif gradient_optimizer == 1:
                p -= eta*derive_p
                q -= eta*derive_q
                
            elif gradient_optimizer == 2 or gradient_optimizer == 3:
                v_p_new = gamma*v_p_old + eta_1*derive_p
                v_q_new = gamma*v_q_old + eta_1*derive_q
                
                p -= v_p_new
                q -= v_q_new
                
                v_p_old = v_p_new
                v_q_old = v_q_new
                        
        ssd_arr.append(new_ssd)

    return registered_images, ssd_arr  # Correct return statement

def register_rotation_ssd(I, J, gradient_optimizer, bins):
    registered_images = []
    ssd_arr = []
    
    # Initialize translation parameters
    theta = math.pi/180
    is_stop = False
    count = 0
    eta = 10**(-10)
    gamma = 0.9
    v_t_old = 0
    
    registered_images.append(J)
    ssd_arr.append(tools.math.ssd(I, J))
    h, w = I.shape
    
    while not is_stop and count <= 100:
        count += 1
        
        if gradient_optimizer != 3:
            new_J, transformed_points_2d = rotate_image(J, theta)
        else:
            new_J, transformed_points_2d = rotate_image(J, theta-gamma*v_t_old)

        registered_images.append(new_J)
        new_ssd = tools.math.ssd(I, new_J)
        x_coordinate = transformed_points_2d[:, 0].reshape(h, w)
        y_coordinate = transformed_points_2d[:, 1].reshape(h, w)

        if has_converged_or_plateaued(ssd_arr, new_ssd):
            is_stop = True
        else:
            gradient_x, gradient_y = np.gradient(new_J)
            derive_theta = np.sum((new_J-I)*(-gradient_x*y_coordinate+gradient_y*x_coordinate))

            if derive_theta == 0:
                is_stop = True
            else:
                if gradient_optimizer == 0:
                    theta -= np.sign(derive_theta)*math.pi/180
                
                elif gradient_optimizer == 1:
                    theta -= eta*derive_theta
                    
                elif gradient_optimizer == 2 or gradient_optimizer == 3:
                    v_t_new = gamma*v_t_old + eta_2*derive_theta
                    theta -= v_t_new
                    v_t_old = v_t_new
                        
        ssd_arr.append(new_ssd)

    return registered_images, ssd_arr  # Correct return statement

def register_rigid_ssd(I, J, gradient_optimizer, bins):
    registered_images = []
    ssd_arr = []
    
    # Initialize translation parameters
    p, q = 0, 0
    theta = math.pi/180
    is_stop = False
    count = 0
    eta_1 = 10**(-6)
    eta_2 = 10**(-10)
    gamma = 0.9
    v_p_old, v_q_old, v_t_old = 0, 0, 0
    
    registered_images.append(J)
    ssd_arr.append(tools.math.ssd(I, J))
    h, w = I.shape

    while not is_stop and count <= 100:
        count += 1
        
        if gradient_optimizer != 3:
            new_J, transformed_points_2d = ridgid_image(J, p, q, theta)
        else:
            new_J, transformed_points_2d = ridgid_image(J, p-gamma*v_p_old, q-gamma*v_q_old, theta-gamma*v_t_old)
            
        registered_images.append(new_J)
        new_ssd = tools.math.ssd(I, new_J)
        x_coordinate = transformed_points_2d[:, 0].reshape(h, w)
        y_coordinate = transformed_points_2d[:, 1].reshape(h, w)

        if has_converged_or_plateaued(ssd_arr, new_ssd):
            is_stop = True
        else:
            gradient_x, gradient_y = np.gradient(new_J)
            derive_theta = np.sum((new_J - I)*(-gradient_x*y_coordinate+gradient_y*x_coordinate))
            derive_p = np.sum((new_J - I) * gradient_x)
            derive_q = np.sum((new_J - I) * gradient_y)
            
            if gradient_optimizer == 0:
                p -= np.sign(derive_p)
                q -= np.sign(derive_q)
                theta -= np.sign(derive_theta)*math.pi/180
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

        ssd_arr.append(new_ssd)

    return registered_images, ssd_arr  # Correct return statement
    