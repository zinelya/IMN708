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
from tools import io, display, register

def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)

    p.add_argument('in_image_1',
                   help='Input image I.')
    p.add_argument('in_image_2',
                   help='Input image J.')
    p.add_argument('register_method', type=int, default=0,
                   help='Register type (0: translation, 1: rotation, 2: rigid (translation+rotation))')
    p.add_argument('--gradient_optimizer', type=int, default=0,
                        help='Gradient descent optimizer: 0-Sign of gradient, 1-Maginitude of gradient, 2-Momentum (for Rigid transformation), 3-NAG (for Rigid transformation)')
    p.add_argument('--eta_t', type=float, default=10**-6,
                        help='Learning rate for gradient descent in translation')
    p.add_argument('--eta_r', type=float, default=10**-10,
                        help='Learning rate for gradient descent in rotation')
    p.add_argument('--n_iterations', type=int, default=1000,
                        help='Maximum number of iterations')
    p.add_argument('--resize_factor', type=int, default=1,
                        help='Desampling ratio for multi-resolution for ridgid transformation')
    p.add_argument('--gaussian_sigma', type=int, default=0,
                        help='Gaussian sigma for denoising for ridgid transformation')
    p.add_argument('--n_last', type=int, default=70,
                        help='Number of last n unchanged SSD ')
    p.add_argument('--convergence_value', type=float, default=100,
                        help='Minimum value of ssd for convergence')
    
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Read arguments
    data1 = io.image_to_data(args.in_image_1)
    data2 = io.image_to_data(args.in_image_2)
    register_method = args.register_method
    gradient_optimizer = args.gradient_optimizer
    eta_t = args.eta_t
    eta_r = args.eta_r
    n_iterations = args.n_iterations
    convergence_value = args.convergence_value
    resize_factor = args.resize_factor
    gaussian_sigma = args.gaussian_sigma
    n_last = args.n_last
        
    if register_method == 0:  
        registered_images, ssd_arr = register.register_translation_ssd(data1, data2, gradient_optimizer, eta_t, n_iterations, convergence_value, n_last)
        # register 'data/BrainMRI_1.jpg' 'data/BrainMRI_2.jpg' 0 --gradient_optimizer 0 --eta_t 1
        # register 'data/BrainMRI_1.jpg' 'data/BrainMRI_2.jpg' 0 --gradient_optimizer 1
    elif register_method == 1:
        registered_images, ssd_arr = register.register_rotation_ssd(data1, data2, gradient_optimizer, eta_r, n_iterations, convergence_value, n_last)
        # register 'data/BrainMRI_1.jpg' 'data/BrainMRI_3.jpg' 1 --gradient_optimizer 0 --eta_r 0.0087 
        # register 'data/BrainMRI_1.jpg' 'data/BrainMRI_3.jpg' 1 --gradient_optimizer 1 
    elif register_method == 2:
        registered_images, ssd_arr, p, q, theta = register.register_rigid_ssd(data1, data2, gradient_optimizer, eta_t, eta_r, n_iterations, convergence_value, resize_factor, gaussian_sigma, n_last)
        # register 'data/BrainMRI_1.jpg' 'data/BrainMRI_2.jpg' 2 --gradient_optimizer 0 --eta_t 1
        # register 'data/BrainMRI_1.jpg' 'data/BrainMRI_3.jpg' 2 --gradient_optimizer 0 --eta_r 0.0087
        # register 'data/BrainMRI_1.jpg' 'data/BrainMRI_2.jpg' 2 --gradient_optimizer 2 --resize_factor 2
        # register 'data/BrainMRI_1.jpg' 'data/BrainMRI_3.jpg' 2 --gradient_optimizer 2 --resize_factor 2
        # register 'data/BrainMRI_1.jpg' 'data/BrainMRI_4.jpg' 2 --gradient_optimizer 2 --resize_factor 2
        # register 'data/BrainMRI_1.jpg' 'data/BrainMRI_4.jpg' 2 --gradient_optimizer 3 --resize_factor 2

    display.display_registration(data1, registered_images, ssd_arr, gradient_optimizer)
        
        
if __name__ == "__main__":
    main()