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


def display_stats(data, bins, title='', min_range=None, max_range=None, resolution=None, vox=None, Michelson=None, RMS=None, std_bg=None, SNR=None):
    """
    Display the histogram of the image data.
    :param data: Numpy array of the image data.
    :param bins: Number of bins for the histogram.
    :param title: Title of the histogram plot.
    """
    # Apply the range filtering if min_range and max_range are provided
    if min_range is not None:
        data = data[data >= min_range]  # Keep data above or equal to min_range
    if max_range is not None:
        data = data[data <= max_range]
    
    # Create the histogram
    n, bins, patches = plt.hist(data.flatten(), bins=bins, density=True, edgecolor='black')
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    # Collect the statistics in a string
    stats_text = ""
    if resolution is not None:
        stats_text += f"Resolution: {resolution[0]:.2f} x {resolution[1]:.2f} x {resolution[2]:.2f}\n"
    if vox is not None:
        stats_text += f"taille des voxels: {vox[0]} x {vox[1]} x {vox[2]}\n"
    if Michelson is not None:
        stats_text += f"Michelson: {Michelson:.2f}\n"
    if RMS is not None:
        stats_text += f"RMS: {RMS:.2f}\n"
    if std_bg is not None:
        stats_text += f"Ecart-type du background: {std_bg:.2f}\n"
    if SNR is not None:
        stats_text += f"SNR: {SNR:.2f}\n"

    # Plot the stats text in a box to the right of the plot
    plt.gcf().text(0.75, 0.6, stats_text, fontsize=12, bbox=dict(facecolor='blue', alpha=0.5))

    # Adjust layout to make room for the text box
    plt.subplots_adjust(right=0.7)

    # Show the plot
    plt.show()
