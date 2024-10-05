#! /usr/bin/env python

"""
Description of what the script does
"""

import argparse
import tools.display
import tools.io
import tools.denoise

def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)

    p.add_argument('in_image', help='Input image.')
    p.add_argument('denoise_method', type=int, default=0,
                   help='Denoise algorithm (0: non-local means, 1: gaussian, 2: anisotropic, 3: median, 4: bilateral).')
    p.add_argument('--output_dir', help='Directory to save the output file.', default='.')
    p.add_argument('--axe', type=int, default=0,
                   help='Axis to display image after denoising (Sagittal 0, Coronal 1, Axial 2)')
    p.add_argument('--sigma', type=float, default=1,
                   help='Sigma value for gaussian denoising')
    p.add_argument('--patch_size', type=int, default=5,
                   help='Size of patches used for non-local means, bilateral, median denoising')
    p.add_argument('--patch_distance', type=int, default=5,
                   help='Maximal distance in pixels where to search patches used for non-local means denoising')
    p.add_argument('--h', type=int, default=30,
                   help='Cut-off distance (filter strength) for non-local means denoising')
    p.add_argument('--sigma_color', type=float, default=None,
                   help='Standard deviation of the intensity differences (color space smoothing) for bilateral denoising')
    p.add_argument('--sigma_spatial', type=float, default=1,
                   help='Standard deviation for range distance. A larger value results in averaging of pixels with larger spatial differences for bilateral denoising')
    p.add_argument('--n', type=float, default=10,
                   help='Number of iterations for anisotropic diffusion denoising')
    p.add_argument('--kappa', type=float, default=50,
                   help='Conductance parameter for anisotropic diffusion')
    p.add_argument('--gamma', type=float, default=0.1,
                   help='Step size for anisotropic diffusion')
    return p

def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    in_image = args.in_image
    denoise_method = args.denoise_method
    output_dir = args.output_dir
    axe = args.axe
    patch_size = args.patch_size
    sigma = args.sigma
    h = args.h
    patch_distance = args.patch_distance
    sigma_color = args.sigma_color
    sigma_spatial = args.sigma_spatial
    n_iter = args.n
    kappa = args.kappa
    gamma = args.gamma

    image = tools.io.check_valid_image(in_image)
    
    if image:
        data = tools.io.reorient_data_rsa(image)
        affine = image.affine  # You will need this for saving the NIfTI file
        voxel_sizes = image.header.get_zooms()

        method_name = ''
        parameters = []

        # Switch case for denoise methods
        if denoise_method == 0:  # non-local means
            denoised_image = tools.denoise.denoise_non_local_means(data, h, patch_size, patch_distance)
            method_name = 'nl_means'
            parameters = [h, patch_size, patch_distance]
        elif denoise_method == 1:  # gaussian
            denoised_image = tools.denoise.denoise_gaussian(data, sigma)
            method_name = 'gaussian'
            parameters = [sigma]
        elif denoise_method == 3:  # median
            denoised_image = tools.denoise.denoise_median(data, patch_size)
            method_name = 'median'
            parameters = [patch_size]
        elif denoise_method == 4:  # bilateral
            denoised_image = tools.denoise.denoise_bilateral(data, patch_size, sigma_color, sigma_spatial)
            method_name = 'bilateral'
            parameters = [patch_size, sigma_color, sigma_spatial]
        elif denoise_method == 5:  # anisotropic diffusion
            denoised_image = tools.denoise.denoise_anisotropic_diffusion(data, niter=n_iter, kappa=kappa, gamma=gamma)
            method_name = 'anisotropic_diffusion'
            parameters = [n_iter, kappa, gamma]

        # Display the denoised image
        tools.display.display_image(denoised_image, voxel_sizes, axe)

        # Save the denoised image
        tools.io.save_nifti_image(denoised_image, affine, in_image, method_name, parameters, output_dir)

if __name__ == "__main__":
    main()
