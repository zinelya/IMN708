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
