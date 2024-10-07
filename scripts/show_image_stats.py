#!/usr/bin/env python

"""
This script displays intensity histograms and statistics
of the entire NIfTI image or specific regions of interest (RoIs).

Features:
- Display an intensity histogram of the input image with configurable bins, range.
- Support for both 2D, 3D and 4D images.
- Optionally, display histograms for specific labels (regions of interest) defined in a separate label mask. (e.g t1_labels.nii.gz for t1.nii.gz)
- Calculate image quality metrics such as standard deviation of the background, mean intensity for each RoI, 
  and SNR (Signal-to-Noise Ratio).

Usage:
------
Example of running the script:

    stats <image> --labels <label_image> --bins 100 --min_range 10 --max_range 500 

Parameters:
-----------
in_image: str
    Path to the input image.
--labels: str, optional
    Path to a label image, used to display histograms for each region of interest (RoI). 
    If provided, the histogram will be computed for each unique label in the image.
--bins: int, optional
    Number of bins to use for the intensity histogram. Default is 100.
--min_range: float, optional
    Minimum intensity value for the histogram range.
--max_range: float, optional
    Maximum intensity value for the histogram range.
"""


import argparse
import nibabel as nib
import numpy as np
import tools.display
import tools.math


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)

    p.add_argument('in_image',
                   help='Input image.'),
    p.add_argument('--labels', type=str, default='',
                        help='If labels is set, show histogram for each label.')
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


    vox = tuple(float(x) for x in image.header.get_zooms())
    taille = image.header.get_data_shape()
    unit = image.header.get_xyzt_units()
    michelson = tools.math.michelson(data)
    RMS = tools.math.rms(data)
    
    if data.ndim == 4 :
        size_4d = data.shape[3]
        tools.display.display_stats(data, args.bins, 'Intensity Histogram of the Image', args.min_range, args.max_range, taille=taille, voxel_size=vox, size_4d=size_4d, unit=unit, Michelson=michelson, RMS=RMS)
    else : 
        # If the image is 4d, display the size of the 4th dimention
        tools.display.display_stats(data, args.bins, 'Intensity Histogram of the Image', args.min_range, args.max_range, taille=taille, voxel_size=vox, unit=unit, Michelson=michelson, RMS=RMS)

    # If a mask is provided, plot histograms for each segment
    if args.labels:
        labels = nib.load(args.labels)
        labels_data = labels.get_fdata()       
        # Extract unique region labels from the mask (excluding 0, which is background)
        unique_labels = np.unique(labels_data[labels_data != 0]).astype(int)
        #std_bg = 0 
        
        # Iterate over each unique label (region of interest) in the mask
        for label in unique_labels:
            # Extract the data corresponding to the current RoI (label)
            roi_data = data[labels_data == label]

            if label == 1:
                # For label 1 (assumed to be the background), calculate the standard deviation
                std_bg = np.std(roi_data)
                mean_bg = np.mean(roi_data)
                tools.display.display_stats(roi_data, args.bins, 'Intensity Histogram of the background', args.min_range, args.max_range, std_bg=std_bg, mean=mean_bg)
            else:
                # For other labels, calculate SNR
                mean_roi = np.mean(roi_data)
                snr = mean_roi / (std_bg + 1e-6)
                std = np.std(roi_data)
                tools.display.display_stats(roi_data, args.bins, f'Intensity Histogram for RoI {label}', args.min_range, args.max_range,std_bg=std_bg, mean=mean_roi, std=std, SNR=snr)


if __name__ == "__main__":
    main()
