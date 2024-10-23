#! /usr/bin/env python

"""
This script reads two input images (I and J) and performs image comparison calculations:
1. It loads the images from their file paths.
2. It checks if the dimensions of the two images match.
3. If the dimensions are the same, it calculates the following metrics:
    - SSD (Sum of Squared Differences)
    - CR (Correlation Coefficient)
    - IM (Mutual Information)
    
The script outputs the calculated values of these metrics for the given image pair.

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
import tools.math
import numpy as np
from PIL import Image


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)

    p.add_argument('in_image_1',
                   help='Input image I.')
    p.add_argument('in_image_2',
                   help='Input image J.')
    p.add_argument('--bins', type=int, default=100,
                        help='Number of bins')
    
    return p

def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Read arguments
    in_image_1 = args.in_image_1
    in_image_2 = args.in_image_2
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
    
        ssd = tools.math.ssd(data_1, data_2)
        cr = tools.math.cr(data_1, data_2)
        im = tools.math.IM(data_1, data_2, bins)
        print('Standard deviation:', ssd)
        print('Correlation coefficient:', cr)
        print('Mutual information:', im)
            
if __name__ == "__main__":
    main()