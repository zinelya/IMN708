import nibabel as nib
import numpy as np

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
    Reorient the image to 'RPS' orientation by flipping along the necessary axes.

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
