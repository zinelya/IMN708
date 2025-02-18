import numpy as np
from skimage.transform import resize
from PIL import Image

"""
    --------------------------------------------------------------------------------------
    --------------------------------------------------------------------------------------
                                             TP2
    --------------------------------------------------------------------------------------
    --------------------------------------------------------------------------------------
"""

def joint_histogram(data1, data2):
    """
    Compute and display the joint histogram of two grayscale images.

    Parameters:
    data1 (numpy.ndarray): First grayscale image.
    data2 (numpy.ndarray): Second grayscale image.
    
    Returns:
    numpy.ndarray: Joint histogram of the two images.
    """
    flat_img1 = data1.flatten()
    flat_img2 = data2.flatten()

    # Initialize the joint histogram
    joint_hist = np.zeros((data1.max()+1, data2.max()+1), dtype=np.int32)

    # Loop over the flattened arrays and populate the joint histogram
    for i in range(len(flat_img1)):
        joint_hist[flat_img1[i], flat_img2[i]] += 1

    #Question 1)b)
    assert np.sum(joint_hist) == data1.size == data2.size

    return joint_hist



def ssd_joint_hist(joint_hist, bin_centers_1, bin_centers_2):
    """
    Calculate the Sum of Squared Differences (SSD) between two images based on the joint histogram.
    
    Parameters:
    joint_hist (numpy.ndarray): Joint histogram of the two images.
    bin_centers_1 (numpy.ndarray): Array of bin centers for the first image intensities.
    bin_centers_2 (numpy.ndarray): Array of bin centers for the second image intensities.
    
    Returns:
    float: The sum of squared differences between I and J.
    """
    # Reshape bin centers to enable broadcasting
    i_vals = bin_centers_1[:, np.newaxis]
    j_vals = bin_centers_2[np.newaxis, :]
    # Compute the squared difference (i - j)^2 using broadcasting
    squared_diff = (i_vals - j_vals) ** 2 
    ssd_value = np.sum(joint_hist * squared_diff)
    
    return ssd_value

def ssd(I, J):
    """Calculate the Sum of Squared Differences (SSD) between two images."""
    return np.sum((I.astype(float) - J.astype(float)) ** 2)


def cr(joint_hist):
    """
    Calculate the Correlation Coefficient (CR) between two images based on the joint histogram.
    Parameters:
    joint_hist (numpy.ndarray): Joint histogram of the two images.
    
    Returns:
    float: The Correlation Coefficient between I and J.
    """
    # Normalize joint histogram
    joint_hist = joint_hist / np.sum(joint_hist)
    
    i_vals = np.arange(joint_hist.shape[0])[:, None]  # Column vector for intensities in I
    j_vals = np.arange(joint_hist.shape[1])[None, :]  # Row vector for intensities in J

    # Calculate weighted means
    mean_I = np.sum(i_vals * joint_hist)
    mean_J = np.sum(j_vals * joint_hist)

    # Calculate numerator
    numerator = np.sum(joint_hist * (i_vals - mean_I) * (j_vals - mean_J))

    # Calculate denominator
    var_I = np.sum(joint_hist * (i_vals ** 2)) - mean_I ** 2
    var_J = np.sum(joint_hist * (j_vals ** 2)) - mean_J ** 2
    denominator = np.sqrt(var_I * var_J)

    if denominator == 0:  # To prevent division by zero
        return 0

    return numerator / denominator


def IM(joint_hist):
    """
    Calculate the Mutual Information (MI) between two images.
    Parameters:
    joint_hist (numpy.ndarray): Joint histogram of the two images.
    
    Returns:
    float: The Mutual Information between I and J.
    """
    # Normalize joint histogram
    joint_prob = joint_hist / np.sum(joint_hist)
    
    # Marginal probabilities for each image
    p_i = np.sum(joint_prob, axis=1, keepdims=True)
    p_j = np.sum(joint_prob, axis=0, keepdims=True)

    # Avoid 0 by masking
    mask = joint_prob > 0

    mi = np.sum(joint_prob[mask] * np.log(joint_prob[mask] / (p_i @ p_j)[mask]))

    return mi

def initialize_grid_points(width, depth, height, init_x, init_y, init_z):
    # Create a 3D grid of points
    x_vals = np.linspace(init_x, init_x + width - 1, width)
    y_vals = np.linspace(init_y, init_y + depth - 1, depth)
    z_vals = np.linspace(init_z, init_z + height - 1, height)
    
    # Create meshgrid for all combinations of x, y, z
    X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')
    
    # Reshape to create N*3 array where each row is a 3D point [x, y, z]
    points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    return points

"""
    --------------------------------------------------------------------------------------
    --------------------------------------------------------------------------------------
                                             TP1
    --------------------------------------------------------------------------------------
    --------------------------------------------------------------------------------------
"""

def michelson(data):
    """
    Computes the Michelson contrast of the image data.
    
    Parameters
    ----------
    data: numpy array
        Numpy array of the image.
    
    Returns
    -------
    Michelson: float
        Michelson contrast value of the image data.
    """
    Min = np.min(data[np.nonzero(data)])
    Max = np.max(data)
    M = (Max - Min)/(Max + Min)
    return M

def rms(data):    
    """
    Computes the RMS contrast of the image data.
    
    Parameters
    ----------
    data: numpy array
        Numpy array of the image.
    
    Returns
    -------
    RMS: float
        RMS contrast value of the image data.
    """
    RMS = np.std(data)
    return RMS

def mIP(data, axe) :
    """
    Computes the minimum Intensity Projection (mIP) along the given axis.
    
    Parameters
    ----------
    data: numpy array
        Numpy array of the image.
    axe: 0 or 1 or 2
        Projection axis (0: Sagittal, 1: Coronal, 2: Axial).   

    Returns
    -------
    mIP: numpy array
        mIP of the image data.
    """    
    mIP = np.min(data, axis=axe)
    return mIP
 
def calc_projection(data, axe, minmax, start_idx=None, end_idx=None):
    """
    Computes the Min/Maximum Intensity Projection (mIP/MIP) along the given axis.
    
    Parameters
    ----------
    data : numpy array
        Numpy array of the image.
    axe: 0 or 1 or 2
        Projection axis (0: Sagittal, 1: Coronal, 2: Axial).   
    minmax: 0 or 1
        Min projection 0, Max projection 1. 
    start_idx: int (optional)
        Starting index for slice range.
    end_idx: int (optional)
        Ending index for slice range.

    Returns
    -------
    projected_data : numpy array
        mIP/MIP of the image data.
    """

    # Extract the relevant slice range along the specified axis
    sliced_data = np.take(data, range(start_idx, end_idx), axis=axe)
    
    # Compute max or min projection along the axis
    if minmax == 1:
        projected_data = np.max(sliced_data, axis=axe)
    else:
        projected_data = np.min(sliced_data, axis=axe)

    # Expand dimensions to match the original shape
    projected_data = np.expand_dims(projected_data, axis=axe)

    return projected_data

def adjust_image_to_shape(image_1, image_2):
    """
    Adjust the size of image_2 to match the shape of image_1 by resizing.

    Parameters:
    - image_1: The reference image (numpy array), whose shape will be used to resize image_2.
    - image_2: The image (numpy array) to be resized to match image_1's shape.

    Returns:
    - image_2_resized: A resized version of image_2 with the same shape as image_1.
    
    The function uses anti-aliasing to ensure smooth resizing of image_2 to match
    the dimensions of image_1.
    """
    image_1_shape = image_1.shape  # Get the shape of image_1
    image_2_resized = resize(image_2, image_1_shape, anti_aliasing=True)
    
    return image_2_resized

def desample_image(image, scale):
    """
    Reduces the resolution of the input image by a specified scaling factor.

    Parameters:
    - image (np.array): The input image to be downscaled, represented as a NumPy array.
    - scale (int): The scaling factor by which the image resolution will be reduced. A scale of 2 will reduce the image size by half.

    Returns:
    - np.array: The downscaled image as a NumPy array with reduced dimensions.
    """
    img = Image.fromarray(image.astype(np.uint8))  # Convert to uint8 for PIL
    h, w = image.shape
    desampled_img = img.resize((w // scale, h // scale), Image.LANCZOS)
    data = np.array(desampled_img).astype(int)
    return data
