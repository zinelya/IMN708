#! /usr/bin/env python
"""
This script displays intensity histograms and statistical comparison of two grayscale images.

Features:
- Display an intensity histogram for each input image with configurable bins.
- If images are normalized (values between 0 and 1), they are rescaled before histogram calculation.

Usage:
------
Example of running the script:

    compare_images <in_image_1> <in_image_2> --bins <num_bins>

Parameters:
-----------
in_image_1: str
    Path to the first input image.
in_image_2: str
    Path to the second input image.
--bins: int, optional
    Number of bins to use for the intensity histogram and joint histogram. Default is 256.
--plt: flag, optional
    If specified, uses Matplotlib for displaying the histograms (default is False).   
"""

import argparse
from tools import io, display

def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)

    p.add_argument('in_image_1',
                   help='Input image 1.'),
    p.add_argument('in_image_2',
                   help='Input image 2.')
    p.add_argument('--bins', type=int, default=256,
                   help='number of bins.')
    p.add_argument('--plt',action='store_true', 
                   help='display using matplotlib.')
    return p

def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    

    data1 = io.image_to_data(args.in_image_1)
    data2 = io.image_to_data(args.in_image_2)
    if args.plt :
        display.plt_display_joint_hist(data1 ,data2 ,bins=args.bins)
    else :     
        display.display_joint_hist(data1 ,data2 ,bins=args.bins)

if __name__ == "__main__":
    main()
