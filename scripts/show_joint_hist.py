import argparse
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt



def joint_histogram(img1, img2,bins):
    """
    Compute and display the joint histogram of two grayscale images by concatenating 
    the flattened images and using a single loop to count the pixel pairs.

    Parameters:
    img1 (numpy.ndarray): First grayscale image.
    img2 (numpy.ndarray): Second grayscale image.
    bins (int): Number of intensity bins for the histogram (default 256 for 8-bit images).
    
    Returns:
    numpy.ndarray: Joint histogram of the two images.
    """
    #check if the two images have the same shape
    #flatten the 2d nparrays and concat
    # Flatten the images
    flat_img1 = img1.flatten()
    flat_img2 = img2.flatten()

    # Initialize the joint histogram
    joint_hist = np.zeros((img1.max()+1, img2.max()+1), dtype=np.int32)

    # Loop over the flattened arrays to populate the joint histogram
    for i in range(len(flat_img1)):
        joint_hist[flat_img1[i], flat_img2[i]] += 1

    joint_hist_log = np.log1p(joint_hist)

    return joint_hist_log, flat_img1, flat_img2

# Example usage:
# img1 and img2 are two grayscale images of the same size
# joint_hist = joint_histogram_seaborn_concat(img1, img2)



def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)

    p.add_argument('in_image_1',
                   help='Input image 1.'),
    p.add_argument('in_image_2',
                   help='Input image 2.')
    return p

def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    
    bins = 256

    data1 = mpimg.imread(args.in_image_1)
    data2 = mpimg.imread(args.in_image_2)


    joint_hist, flat_img1, flat_img2=joint_histogram(data1, data2, bins=256)

    # Create a figure with 3 subplots: 1 for the joint histogram and 2 for the individual histograms
    fig, axs = plt.subplots(2, 2, figsize=(10,10), gridspec_kw={'width_ratios': [1, 4], 'height_ratios': [4, 1]})

    # Plot the individual histogram of Image 1 in the top-left subplot
    axs[1, 1].hist(flat_img2, bins=bins, color='black', alpha=0.7)
    axs[1, 1].set_title('Histogram of Image 2')
    



    # Plot the joint histogram heatmap in the top-right subplot using imshow
    cax = axs[0, 1].imshow(joint_hist, cmap='hot', origin='lower', aspect='auto')
    fig.colorbar(cax, ax=axs[0, 1])  # Add color bar to the joint histogram
    axs[0, 1].set_title('Joint Histogram Heatmap with Logarithmic Scaling')
    axs[0, 1].set_xlabel('Intensity Value of Image 2')
    axs[0, 1].set_ylabel('Intensity Value of Image 1')

    # Plot the individual histogram of Image 2 in the bottom-left subplot, rotated by 90 degrees
    axs[0, 0].hist(flat_img1, bins=bins, color='black', alpha=0.7, orientation='horizontal')
    axs[0, 0].set_title('Histogram of Image 1')
    axs[0, 0].invert_xaxis()



    # Leave the bottom-right subplot empty for future metrics
    axs[1, 0].axis('off')
    axs[1, 0].set_title('Metrics (To be added later)')

    # Adjust layout to prevent overlapping titles and labels
    plt.tight_layout()

    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()
