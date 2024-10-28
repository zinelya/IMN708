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
import scipy.ndimage as ndimage

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
    p.add_argument('--gradient_optimizer', type=int, default=0,
                        help='Gradient descent optimizer: 0-Fix step, 1-Learning rate, 2-Momentum, 3-NAG')
    p.add_argument('--n_iterations', type=int, default=1000,
                        help='Maximum number of iterations')
    p.add_argument('--resize_factor', type=int, default=1,
                        help='Desampling ratio for multi-resolution')
    p.add_argument('--gaussian_sigma', type=int, default=0,
                        help='Gaussian sigma for denoising')
    p.add_argument('--n_elements', type=int, default=20,
                        help='Number of last n unchanged SSD ')
    p.add_argument('--convergence_value', type=float, default=2*10**7,
                        help='Minimum value of ssd for convergence')
    
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Read arguments
    in_image_1 = args.in_image_1
    in_image_2 = args.in_image_2
    register_method = args.register_method
    bins = args.bins
    gradient_optimizer = args.gradient_optimizer
    n_iterations = args.n_iterations
    convergence_value = args.convergence_value
    resize_factor = args.resize_factor
    gaussian_sigma = args.gaussian_sigma
    n_elements = args.n_elements

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
            registered_images, ssd_arr = tools.register.register_translation_ssd(data_1, data_2, gradient_optimizer, n_iterations, convergence_value, bins)
        elif register_method == 1:
            registered_images, ssd_arr = tools.register.register_rotation_ssd(data_1, data_2, gradient_optimizer, n_iterations, convergence_value, bins)
        elif register_method == 2:
            registered_images, ssd_arr, p, q, theta = tools.register.register_rigid_ssd(image_1, image_2, gradient_optimizer, n_iterations, convergence_value, resize_factor, gaussian_sigma, n_elements, bins)
        
        tools.display.display_registration(data_1, registered_images, ssd_arr)
        
        
if __name__ == "__main__":
    main()