#!/usr/bin/env python

import nibabel as nib
import numpy as np
import argparse
import os

def apply_mask_dmri(image_path, mask_path, output_path):
    """
    Applies a binary mask to a dMRI NIfTI image (4D) and saves the result.

    Args:
        image_path (str): Path to the input dMRI NIfTI image.
        mask_path (str): Path to the binary mask NIfTI image.
        output_path (str): Path to save the masked dMRI NIfTI image.
    """
    # Load the NIfTI image and mask
    image_nii = nib.load(image_path)
    mask_nii = nib.load(mask_path)
    
    # Get the image and mask data
    image_data = image_nii.get_fdata()
    mask_data = mask_nii.get_fdata()
    
    # Check if the mask dimensions match the first 3 dimensions of the image
    if image_data.shape[:3] != mask_data.shape:
        raise ValueError("The dimensions of the mask do not match the spatial dimensions of the image.")
    
    # Ensure the mask is binary
    binary_mask = (mask_data > 0).astype(np.float32)
    
    # Apply the mask to each direction (4th dimension)
    masked_data = image_data * binary_mask[..., np.newaxis]
    
    # Create a new NIfTI image for the masked data
    masked_nii = nib.Nifti1Image(masked_data, image_nii.affine, image_nii.header)
    
    # Save the masked image
    nib.save(masked_nii, output_path)
    print(f"Masked image saved to {output_path}")

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Apply a binary mask to a 4D dMRI NIfTI image.")
    parser.add_argument("image", help="Path to the input dMRI NIfTI image")
    parser.add_argument("mask", help="Path to the binary mask NIfTI image")
    parser.add_argument("output", help="Path to save the masked dMRI NIfTI image")
    
    args = parser.parse_args()
    
    # Apply the mask
    apply_mask_dmri(args.image, args.mask, args.output)
