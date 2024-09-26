import matplotlib.pyplot as plt
import numpy as np
import scipy


def michelson(image):
    "Calcule le contraste de Michelson d'une image"
    Min = np.min(image[np.nonzero(image)])
    Max = np.max(image)
    print(np.min(image))
    print(np.max(image))
    M = (Max - Min)/(Max + Min)
    return M

def rms(image):
    "Calcule le contraste Root-Mean-Square d'une image"
    RMS = np.std(image)
    return RMS

#TODO
def snr(image) :
    "how to crop : https://github.com/scilus/scilpy/blob/master/scilpy/image/volume_operations.py"
    noise = image[2:20, 3:40, 3]
    bruit = np.std(noise)
    signal = image[50:70, 55:80, 60:65]
    mean = np.mean(signal)
    snr = mean/signal

    return snr

def mIP(image, axe) :    
    if 'x' in axe:
        mIP = np.min(image, axis=0)

    if 'y' in axe:
        mIP = np.min(image, axis=1)

    if 'z' in axe:
        mIP = np.min(image, axis=2)
    fig, ax = plt.subplots()
    ax.imshow(mIP, cmap='gray')
    ax.set_title("mIP")
    plt.show()   


def MIP(image,axe) :
    if 'x' in axe:
        mIP = np.max(image, axis=0)

    if 'y' in axe:
        mIP = np.max(image, axis=1)

    if 'z' in axe:
        mIP = np.max(image, axis=2)
    fig, ax = plt.subplots()
    ax.imshow(mIP, cmap='gray')
    ax.set_title("MIP")
    plt.show()   

def entropy(image):
    value,counts = np.unique(image, return_counts=True)
    entropy = scipy.stats.entropy(counts)

    return entropy