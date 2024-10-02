#! /usr/bin/env python

"""
Description of what the script does
"""

import argparse
import tools.utils
import numpy as np
import nibabel as nib
import tools.display
import tools.math


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)

    p.add_argument('in_image',
                   help='Input image.'),
    p.add_argument('--in_labels', type=str, default='',
                        help='If in_labels is set, show histogram for each label.')
    p.add_argument('--bins', type=int, default=100,
                        help='Number of bins')
    p.add_argument('--min_range', type=float, default=None,
                   help='Minimum value for the histogram range.')
    p.add_argument('--max_range', type=float, default=None,
                   help='Maximum value for the histogram range.')


    return p

def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    image = nib.load(args.in_image)
    data = image.get_fdata()
    
    # Extract image data as a NumPy array and rescale if negative value exists (CT scan)
    data = image.get_fdata()
    if np.any(data < 0):
        data = data - np.min(data)

    resolution = tuple(float(x) for x in image.header.get_zooms())
    vox = image.header.get_data_shape()
    michelson = tools.math.michelson(data)
    RMS = tools.math.rms(data)

    tools.display.display_stats(data, args.bins, 'Intensity Histogram of the Image', args.min_range, args.max_range, resolution=resolution, vox=vox, Michelson=michelson, RMS=RMS)

    # If a mask is provided, plot histograms for each segment
    
    if args.in_labels:
        labels = nib.load(args.in_labels)
        labels_data = labels.get_fdata()       
        # Extract unique region labels from the mask (excluding 0, which is background)
        unique_labels = np.unique(labels_data[labels_data != 0]).astype(int)
        std_bg = 0 
        
        # Iterate over each unique label (region of interest) in the mask
        for label in unique_labels:
            # Extract the data corresponding to the current RoI (label)
            roi_data = data[labels_data == label]

            if label == 1:
                # For label 1 (assumed to be the background), calculate the standard deviation
                std_bg = np.std(roi_data)
                tools.display.display_stats(roi_data, args.bins, 'Intensity Histogram of the background', std_bg=std_bg)
                print(f'Standard deviation of intensity (background): {std_bg}')

            else:
                # For other RoIs, calculate the mean intensity and SNR
                mean_roi = np.mean(roi_data)
                snr = mean_roi / (std_bg + 1e-6)  # Avoid division by zero by adding a small value
                tools.display.display_stats(roi_data, args.bins, f'Intensity Histogram for RoI {label}', SNR=snr)
                print(f'RoI {label}, Mean of intensity: {mean_roi}, SNR: {snr}, min intensity: {np.min(roi_data)}, max intensity: {np.max(roi_data)}')


if __name__ == "__main__":
    main()
