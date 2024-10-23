import matplotlib.pyplot as plt
import numpy as np
import scipy

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
    #TODO
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

def convert_img_to_bin(data, bins):
    # Get min/max intensity value of image
    min_val = np.min(data)
    max_val = np.max(data)
    
    step = (max_val - min_val)/bins
    bin_data = np.zeros_like(data)
    
    for i in range(0, data.shape[0]):
        for j in range(0, data.shape[1]):
            bin_th = int((data[i][j]-min_val)/step)
            bin_data[i][j] = bin_th if bin_th < bins else bins - 1

    return bin_data
    
def calc_joint_hist(data_1, data_2, bins):
    bin_data_1 = convert_img_to_bin(data_1, bins)
    bin_data_2 = convert_img_to_bin(data_2, bins)
    
    joint_hist = np.zeros((bins, bins), dtype=int)
    
    for i in range(0, data_1.shape[0]):
        for j in range(0, data_1.shape[1]):
            bin_ij_1 = bin_data_1[i][j]
            bin_ij_2 = bin_data_2[i][j]
            joint_hist[bin_ij_1][bin_ij_2] += 1
    
    return joint_hist

def ssd(I, J):
    """Calculate the Sum of Squared Differences (SSD) between two images."""
    return np.sum((I.astype(float) - J.astype(float)) ** 2)

def cr(I, J):
    """Calculate the Correlation Coefficient (CR) between two images manually."""
    I_flat = I.flatten()
    J_flat = J.flatten()
    
    # Calculate means
    mean_I = np.mean(I_flat)
    mean_J = np.mean(J_flat)
    
    # Calculate numerator and denominator for the Pearson correlation coefficient
    numerator = np.sum((I_flat - mean_I) * (J_flat - mean_J))
    denominator = np.sqrt(np.sum((I_flat - mean_I) ** 2)) * np.sqrt(np.sum((J_flat - mean_J) ** 2))
    
    if denominator == 0:  # To prevent division by zero
        return 0
    
    return numerator / denominator

def IM(I, J, bins=256):
    """Calculate Mutual Information (IM) between two images."""
    # Compute joint histogram
    joint_hist = calc_joint_hist(I, J, bins)
    
    # Normalize to get joint probability distribution
    joint_prob = joint_hist / joint_hist.sum()

    # Compute marginal probabilities
    prob_I = joint_prob.sum(axis=1)
    prob_J = joint_prob.sum(axis=0)

    # Avoid log(0) by adding a small value
    joint_prob += 1e-10
    prob_I += 1e-10
    prob_J += 1e-10

    # Calculate mutual information
    mi = 0.0
    for i in range(joint_prob.shape[0]):
        for j in range(joint_prob.shape[1]):
            if joint_prob[i, j] > 0:
                mi += joint_prob[i, j] * np.log(joint_prob[i, j] / (prob_I[i] * prob_J[j]))

    return mi