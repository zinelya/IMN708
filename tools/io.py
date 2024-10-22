import nibabel as nib
import numpy as np
import os 
import matplotlib.image as mpimg

"""
    --------------------------------------------------------------------------------------
    --------------------------------------------------------------------------------------
                                             TP1
    --------------------------------------------------------------------------------------
    --------------------------------------------------------------------------------------
"""

def check_valid_image(in_image):
    """
    Checks the validity of the image and expands dimensions if necessary.
    
    Parameters
    ----------
    image: NIfTI image object.
    """
    try:
        # Load image
        image = nib.load(in_image)
        # Determine the number of axes (dimensions)
        num_axes = len(image.shape)
        
        if num_axes < 2:
            print('The number of image dimensions must be greater than 1.')
            return None 
        
        if num_axes == 2:
            affine = image.affine
            data = image.get_fdata()   # Flip the image data along appropriate axes to achieve 'RPS' orientation.
            # Expand along the last axis
            data_3d = np.expand_dims(data, axis=-1)  
            image = nib.Nifti1Image(data_3d, affine)
        
        return image  # Return the original or modified image
    except Exception as e:
        print('Error in loading image:', e)
        return None
    
def reorient_data_rsa(image):
    """
    Reorient the image to 'RAS' orientation by flipping along the necessary axes.

    Parameters
    ----------
    image: NIfTI image object.

    Returns
    -------
    Reoriented data array.
    """
    
    data = image.get_fdata()
    x, y, z = nib.aff2axcodes(image.affine)
    
    if x != 'R':
        data = np.flip(data, axis=0)
    if y != 'A':
        data = np.flip(data, axis=1)
    if z != 'S':
        data = np.flip(data, axis=2)

    return data

import os
import nibabel as nib

def save_nifti_image(data, affine, input_filename, method_name, parameters, output_dir, residual=False):
    """
    Saves a numpy array as a NIfTI image with the following naming convention:
    - t1.nii.gz denoised with nl_means [h=1, patch_size=5, patch_distance=8] --> t1.nl_means_1_5_8.nii.gz
    - t1.nii.gz denoised with anisotropic_diffusion [niter=10, kappa=50, gamma=0.1] --> t1.anisotropic_diffusion_10_50_0.1.nii.gz

    This function also creates a subdirectory 'denoised_data' inside the specified output directory to store the files.

    Parameters
    ----------
    data : numpy.ndarray
        The denoised image data to be saved.
    affine : numpy.ndarray
        The affine transformation matrix of the initial NIfTI image.
    input_filename : str
        The original input filename from which the base name is extracted.
    method_name : str
        The denoising method name to append to the output file.
    parameters : list
        List of parameters chosen for the method.
    output_dir : str
        The directory where the output file will be saved.
    """
    # Create the denoised_data directory if it doesn't exist
    if residual :
        denoised_dir = os.path.join(output_dir, "residuals")
    else : 
        denoised_dir = os.path.join(output_dir, "denoised_data")
    if not os.path.exists(denoised_dir):
        os.makedirs(denoised_dir)

    # Extract base name and construct the output filename
    base_name = os.path.basename(input_filename).split('.')[0]
    param_str = "_".join(map(str, parameters))  # Convert parameters to string and join with underscores
    output_filename = f"{base_name}_{method_name}_{param_str}.nii.gz"
    
    # Create the full output path within denoised_data directory
    output_path = os.path.join(denoised_dir, output_filename)
    
    # Save the denoised NIfTI image
    nifti_img = nib.Nifti1Image(data, affine)
    nib.save(nifti_img, output_path)

    # Print the path for confirmation
    print(f"Saved: {output_path}")


"""
    --------------------------------------------------------------------------------------
    --------------------------------------------------------------------------------------
                                             TP2
    --------------------------------------------------------------------------------------
    --------------------------------------------------------------------------------------
"""


def split_name_with_nii(filename):
    base, ext = os.path.splitext(filename)

    if ext == ".gz":
        # Test if we have a .nii additional extension
        tmp_base, extra_ext = os.path.splitext(base)

        if extra_ext == ".nii":
            ext = extra_ext + ext
            base = tmp_base

    return base, ext

def image_to_data(filename) :
    basename, extension = split_name_with_nii(filename)
    print(extension)
    if extension in ['.png', '.jpg']:
        data = mpimg.imread(filename)
        if data.ndim == 3:
            data = np.average(data, axis=2)
    elif extension in ['.nii', '.nii.gz']:
        image = check_valid_image(filename)
        data = reorient_data_rsa(image)
    return data

def rescale_and_discretize_image(image, max):
    """
    Rescale an image so that its minimum value becomes 0 and its maximum value becomes 255.
    
    Parameters:
    image (numpy.ndarray): Input image with arbitrary pixel values.
    
    Returns:
    numpy.ndarray: Image rescaled to have pixel values between 0 and 255.
    """
    
    # Get the min and max values of the image
    min_value = image.min()
    max_value = image.max()
    
    # Avoid division by zero in case the image has constant values
    if max_value > min_value:
        # Rescale the image to the range [0, 255]
        rescaled_image = (image - min_value) / (max_value - min_value) * max
    else:
        # If min_value == max_value, the image is constant, set all values to 255
        rescaled_image = np.full_like(image, max)

    # Round and convert to uint8
    rescaled_image = np.round(rescaled_image).astype(np.uint8)
    
    return rescaled_image