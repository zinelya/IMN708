import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
import tools.io
from scipy.ndimage import gaussian_filter
from skimage import restoration

def denoise_gaussian(data, sigma):
    denoised_image = gaussian_filter(data, sigma=sigma)
    return denoised_image

def denoise_non_local_means(data, h, patch_size, patch_distance):
    denoised_image = restoration.denoise_nl_means(data, h=h, patch_size=patch_size, patch_distance=patch_distance, fast_mode=True)
    return denoised_image

def denoise_bilateral(data, win_size, sigma_color, sigma_spatial):
    denoised_image = restoration.denoise_bilateral(data[:, :, 0], win_size=win_size, sigma_color=sigma_color, sigma_spatial=sigma_spatial)
    denoised_image = np.expand_dims(denoised_image, axis=-1)  
    return denoised_image
