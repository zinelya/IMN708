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


def display_histogram(data, bins, title='', min_range=None, max_range=None):
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
    
    plt.hist(data.flatten(), bins=bins, density=True, edgecolor='black')
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()



def histogram(data, bins, min_range, max_range, labels_data=None):

    # Display the overall intensity histogram
    display_histogram(data, bins, 'Intensity Histogram', min_range, max_range)

    # If a mask is provided, plot histograms for each segment
    if labels_data.any():        
        # Extract unique region labels from the mask (excluding 0, which is background)
        unique_labels = np.unique(labels_data[labels_data != 0]).astype(int)
        std_bg = 0 
        
        # Iterate over each unique label (region of interest) in the mask
        for label in unique_labels:
            # Extract the data corresponding to the current RoI (label)
            roi_data = data[labels_data == label]
            display_histogram(roi_data, bins, f'Intensity Histogram for RoI {label}', min_range, max_range)
            
            if label == 1:
                # For label 1 (assumed to be the background), calculate the standard deviation
                std_bg = np.std(roi_data)
                print(f'Standard deviation of intensity (background): {std_bg}')
            else:
                # For other RoIs, calculate the mean intensity and SNR
                mean_roi = np.mean(roi_data)
                snr = mean_roi / (std_bg + 1e-6)  # Avoid division by zero by adding a small value
                print(f'RoI {label}, Mean of intensity: {mean_roi}, SNR: {snr}, min intensity: {np.min(roi_data)}, max intensity: {np.max(roi_data)}')