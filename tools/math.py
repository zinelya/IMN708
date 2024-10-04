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
 


def MIP(data, axe) :
    """
    Computes the Maximum Intensity Projection (MIP) along the given axis.
    
    Parameters
    ----------
    data : numpy array
        Numpy array of the image.
    axe: 0 or 1 or 2
        Projection axis (0: Sagittal, 1: Coronal, 2: Axial).   

    Returns
    -------
    MIP : numpy array
        MIP of the image data.
    """     
    MIP = np.max(data, axis=axe)
    return MIP
