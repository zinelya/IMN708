#!/usr/bin/env python

"""
This script processes diffusion MRI (dMRI) data to compute diffusion tensor imaging (DTI) metrics.

Features:
---------
- Fits the diffusion tensor using a given gradient table.
- Computes standard DTI metrics (FA, MD, AD, RD, RGB, and ADC).
- Applies an optional white matter mask to tensors and peaks before saving.

Usage:
------
    dti_metrics dmri.nii.gz grad_table.txt output_dir/ --tensor_order mrtrix --mask wm_mask.nii.gz

Parameters:
-----------
- dmri (str): 
Path to the input diffusion MRI (dMRI) data in NIfTI format.
- g_tab (str): 
Path to the gradient table file (b-values and b-vectors).
- out_dir (str): 
Path to the output directory where results will be saved.
- --tensor_order (str): 
The tensor fitting convention to use ('fsl' or 'mrtrix'). Default is 'mrtrix'.
- --mask (str): 
Optional path to a white matter mask in NIfTI format. When provided, the mask is applied to tensors and peaks.

Outputs:
--------
The script generates the following files in the specified output directory:
- `tensors.nii.gz`: Diffusion tensor.
- `peaks.nii.gz`: Principal diffusion directions.
- `fa.nii.gz`: Fractional anisotropy.
- `md.nii.gz`: Mean diffusivity.
- `ad.nii.gz`: Axial diffusivity.
- `rd.nii.gz`: Radial diffusivity.
- `rgb.nii.gz`: RGB map scaled by FA.
- `adc.nii.gz`: Apparent diffusion coefficient.
"""

import argparse
import nibabel as nib
import numpy as np
from tools.io import save_nifti_img, read_grad_file
import tools.tensor

def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)
    # Add arguments here
    p.add_argument('dmri', type=str, help='Path to the dmri.')
    p.add_argument('g_tab', type=str, help='Path to the gradient table.')
    p.add_argument('out_dir', type=str, help='Path to the output directory.')
    p.add_argument('--tensor_order', type=str, default = 'mrtrix',
                   help='fsl or mrtrix, default = mrtrix.')
    p.add_argument('--mask', type=str, default = None,
                   help='white matter mask for tensors and peaks.')
    return p

def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    dmri_path = args.dmri
    dmri = nib.load(dmri_path)
    dmri_data = dmri.get_fdata()

    bval, bvecs = read_grad_file(args.g_tab)

    B = tools.tensor.compute_system_matrix(bval, bvecs, args.tensor_order)

    dti = tools.tensor.fit_dti(dmri_data, B)
    evals, _ , peaks = tools.tensor.SVD_dti(dti, args.tensor_order)

    fa, md, ad, rd, rgb, adc = tools.tensor.compute_dti_metrics(dti, evals, peaks, args.tensor_order)

    if args.mask:
        mask_path = args.mask
        mask_img = nib.load(mask_path)
        mask = mask_img.get_fdata().astype(bool)

        dti = np.where(mask[..., np.newaxis], dti, 0)
        peaks = np.where(mask[..., np.newaxis], peaks, 0)

    save_nifti_img(dti, dmri.affine, args.out_dir + 'tensors.nii.gz')
    save_nifti_img(peaks, dmri.affine, args.out_dir +'peaks.nii.gz')
    save_nifti_img(fa, dmri.affine, args.out_dir + 'fa.nii.gz')
    save_nifti_img(rgb, dmri.affine, args.out_dir + 'rgb.nii.gz')
    save_nifti_img(md, dmri.affine, args.out_dir + 'md.nii.gz')
    save_nifti_img(ad, dmri.affine, args.out_dir + 'ad.nii.gz')
    save_nifti_img(rd, dmri.affine, args.out_dir + 'rd.nii.gz')
    save_nifti_img(adc, dmri.affine, args.out_dir + 'adc.nii.gz')

if __name__ == "__main__":
    main()
