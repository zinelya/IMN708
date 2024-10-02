import matplotlib.pyplot as plt
import numpy as np
import scipy


def michelson(image):
    "Calcule le contraste de Michelson d'une image"
    Min = np.min(image[np.nonzero(image)])
    Max = np.max(image)
    M = (Max - Min)/(Max + Min)
    return M

def rms(image):
    "Calcule le contraste Root-Mean-Square d'une image"
    RMS = np.std(image)
    return RMS

def mIP(image, axe) :    
    mIP = np.min(image, axis=axe)
    return mIP
 


def MIP(image, axe) :    
    MIP = np.max(image, axis=axe)
    return MIP

def entropy(image):
    value,counts = np.unique(image, return_counts=True)
    entropy = scipy.stats.entropy(counts)

    return entropy