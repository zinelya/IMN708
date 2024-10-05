import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from scipy.ndimage import gaussian_filter, median_filter
from skimage import restoration
from medpy.filter.smoothing import anisotropic_diffusion
from skimage.restoration import denoise_wavelet

def denoise_median(data, patch_size):
    denoised_image = median_filter(data, size=patch_size)
    return denoised_image

def denoise_gaussian(data, sigma):
    denoised_image = gaussian_filter(data, sigma=sigma)
    return denoised_image

def denoise_non_local_means(data, h, patch_size, patch_distance):
    denoised_image = restoration.denoise_nl_means(data, h=h, patch_size=patch_size, patch_distance=patch_distance, fast_mode=True)
    return denoised_image

def denoise_bilateral(data, patch_size, sigma_color, sigma_spatial):
    # Initialize list to store denoised images for each axis
    combined_data = []

    # Loop through each axis (dimension) of the 3D data
    for dim in range(3):  # Axis 0, 1, and 2
        denoised_data = []

        # Loop through each slice along the current axis
        for i in range(data.shape[dim]):
            if dim == 0:
                # Slicing along axis 0 (shape: [slice_height, slice_width])
                slice_2d = data[i, :, :]
            elif dim == 1:
                # Slicing along axis 1 (shape: [slice_depth, slice_width])
                slice_2d = data[:, i, :]
            else:
                # Slicing along axis 2 (shape: [slice_depth, slice_height])
                slice_2d = data[:, :, i]

            # Apply bilateral filter to the 2D slice
            denoised_slice = restoration.denoise_bilateral(slice_2d, 
                                                           win_size=patch_size, 
                                                           sigma_color=sigma_color, 
                                                           sigma_spatial=sigma_spatial)

            # Expand dimensions to restore the slice in 3D format
            if dim == 0:
                denoised_slice = np.expand_dims(denoised_slice, axis=0)  # Add back the missing axis
            elif dim == 1:
                denoised_slice = np.expand_dims(denoised_slice, axis=1)  # Add back the missing axis
            else:
                denoised_slice = np.expand_dims(denoised_slice, axis=2)  # Add back the missing axis

            # Append the 3D slice to the list
            denoised_data.append(denoised_slice)

        # Stack slices along the current axis to form a denoised 3D image
        denoised_data = np.concatenate(denoised_data, axis=dim)
        combined_data.append(denoised_data)

    # Compute the average of the three 3D denoised images (one for each axis)
    combined_data = np.mean(combined_data, axis=0)

    return combined_data

def denoise_anisotropic_diffusion(data, niter, kappa, gamma):
    denoised_image = anisotropic_diffusion(data, niter, kappa, gamma)
    return denoised_image