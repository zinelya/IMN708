#! /usr/bin/env python

"""
This script reads two input images (I and J) and computes their joint histogram using a specified number of bins.
The images are loaded from their file paths, and their dimensions are checked to ensure they match.
If the images have the same dimensions, the joint histogram is computed and displayed.

Arguments:
in_image_1: str
    Path to the first input image (I).
in_image_2: str
    Path to the second input image (J).
--bins: int, optional
    The number of bins for the joint histogram (default: 100).
"""

import argparse
import tools.display
import tools.register
import numpy as np
from PIL import Image


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)

    p.add_argument('in_image_1',
                   help='Input image I.')
    p.add_argument('in_image_2',
                   help='Input image J.')
    p.add_argument('register_method', type=int, default=0,
                   help='Register type (0: translation, 1: rotation, 2: rigid (translation+rotation))')
    p.add_argument('--bins', type=int, default=100,
                        help='Number of bins')
    
    return p

def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Read arguments
    in_image_1 = args.in_image_1
    in_image_2 = args.in_image_2
    register_method = args.register_method
    bins = args.bins

    # Load images from path
    image_1 = Image.open(in_image_1)
    image_2 = Image.open(in_image_2)
    
    # Get shape of images
    w_1, h_1 = image_1.size
    w_2, h_2 = image_2.size
    
    # Check shapes of image I and J
    if w_1 == w_2 and h_1 == h_2:
        #Load data from images
        data_1 = np.array(image_1).astype(int)
        data_2 = np.array(image_2).astype(int)
        
        if register_method == 0:  
            registered_images, ssd_arr = tools.register.register_translation_ssd(data_1, data_2, bins)
        elif register_method == 1:
            registered_images, ssd_arr = tools.register.register_rotation_ssd(data_1, data_2, bins)
        elif register_method == 2:
            registered_images, ssd_arr = tools.register.register_rigid_ssd(data_1, data_2, bins)
        
        tools.display.display_registration(registered_images, ssd_arr)
        
        
if __name__ == "__main__":
    main()