"""
This script reads two input images (I and J) and computes their joint histogram using a specified number of bins.
The images are loaded from their file paths, and their dimensions are checked to ensure they match.
If the images have the same dimensions, the joint histogram is computed and displayed.
 
Arguments:
fix_image: str
    Path to the first input image (I).
mov_image: str
    Path to the second input image (J).
--bins: int, optional
    The number of bins for the joint histogram (default: 100).
"""
 
import argparse
from tools import io, display, register
import numpy as np
import os
 
def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)
 
    p.add_argument('fix_image',
               help='Path to the first input image (I).')
    p.add_argument('transform_method', type=int, default=0,
                help='Transform method: 0 for translation, 1 for rotation, 2 for rigid, 3 for register_translation_ssd, 4 for register_rotation_ssd and 5 for register_rigid_ssd.')
   
    # Arguments for transformation
    p.add_argument('--p', type=float, default=0,
                   help='Translation distance in x-axis')
    p.add_argument('--q', type=float, default=0,
                   help='Translation distance in y-axis')
    p.add_argument('--theta', type=float, default=0,
                   help='Rotation angle (degree) from x-axis(+) to y-axis(-)')
    p.add_argument('--out_dir', type=str, default='./data',
                   help='Directory of transformed file')
   
    # Arguments for registration
    p.add_argument('--mov_image',
                help='Path to the second input image (J).')
    p.add_argument('--grad_optimizer', type=int, default=0,
                help='Type of gradient descent optimizer: 0 for regular gradient descent, 1 for momentum-based gradient descen, 2 fpr adam-based gradient descent.')
    p.add_argument('--eta_t', type=float, default=10**-7,
                help='Learning rate for gradient descent applied to translation.')
    p.add_argument('--eta_r', type=float, default=10**-11,
                help='Learning rate for gradient descent applied to rotation.')
    p.add_argument('--res_level', type=int, default=0,
                help="""Desampling ratio for multi-resolution rigid transformation. Based on res_level,
                            a resolution hierarchy is constructed [1/2^res_level, 1/2^(res_level-1) to 1/2^0].
                            Example:
                            Level 0: register image at full resolution (1/2^0 = 1).
                            Level 1: register at 1/2 resolution, then full resolution.""")
    p.add_argument('--gauss_sigma', type=float, default=0,
                help='Sigma value for Gaussian denoising in rigid transformation (0 means no denoising).')
    p.add_argument('--n_iter', type=int, default=1000,
                help='Maximum number of iterations for convergence.')
    p.add_argument('--conv_thresh', type=float, default=14000000,
                help='Convergence threshold based on minimum sum of squared differences (SSD) between images.')
    p.add_argument('--max_std', type=float, default=1000,
                help='Maximum standard deviation for SSD over iterations as a convergence criterion.')
    p.add_argument('--n_last', type=int, default=50,
                help='Number of last iterations to evaluate maximum standard deviation of SSD for convergence.')
    p.add_argument('--momentum', type=float, default=0.9,
                help='Momentum value for momentum gradient descent')
 
    return p
 
def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
 
    # Read arguments
    data1 = io.image_to_data(args.fix_image)
    p = args.p
    q = args.q
    theta = args.theta
    out_dir = args.out_dir
    transform_method = args.transform_method
    grad_optimizer = args.grad_optimizer
    eta_t = args.eta_t
    eta_r = args.eta_r
    n_iter = args.n_iter
    conv_thresh = args.conv_thresh
    res_level = args.res_level
    gauss_sigma = args.gauss_sigma
    n_last = args.n_last
    max_std = args.max_std
    momentum = args.momentum
   
    if transform_method <= 2:
        method = ''
        params = []
        if transform_method == 0:  
            new_img = register.translate_image(data1, -p, -q)
            method = 'translation'
            params = [p, q]
        elif transform_method == 1:  
            rad = np.radians(theta)
            new_img, _ = register.rotate_image(data1, -rad)
            method = 'rotation'
            params = [theta]
        else:  
            rad = np.radians(theta)
            new_img, _ = register.rigid_image(data1, -p, -q, -rad)
            method = 'rigid'
            params = [p, q, theta]
       
        display.display_transformed_image(data1, new_img)
        io.save_2d_image(new_img, args.fix_image, method, params, out_dir)
    else:
        data2 = io.image_to_data(args.mov_image)
        mov_img_name = os.path.basename(args.mov_image).split('.')[0]
        ssd_hist, scale_hist, p_hist, q_hist, theta_hist = [], [], [], [], []
       
        if transform_method == 3:  
            grad_optimizer = 0
            reg_images, ssd_hist, p_hist, q_hist = register.register_translation_ssd(data1, data2, eta_t, n_iter, conv_thresh, n_last, max_std)
        elif transform_method == 4:
            grad_optimizer = 0
            reg_images, ssd_hist, theta_hist = register.register_rotation_ssd(data1, data2, eta_r, n_iter, conv_thresh, n_last, max_std)
        else:
            reg_images, ssd_hist, p_hist, q_hist, theta_hist, scale_hist = register.register_rigid_ssd(data1, data2, grad_optimizer, eta_t, eta_r, n_iter, conv_thresh, res_level, gauss_sigma, n_last, max_std, momentum)
        
        display.display_registration(data1, reg_images, ssd_hist, p_hist, q_hist, theta_hist, scale_hist, grad_optimizer, res_level, mov_img_name)
        display.display_joint_hist(data1, data2,bins=50)
        display.display_joint_hist(data1, reg_images[-1],bins=50)
        
if __name__ == "__main__":
    main()
