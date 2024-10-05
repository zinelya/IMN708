#! /usr/bin/env python

"""
This script reads a NIfTI image and computes the Min/Max Intensity Projection (MIP/mIP) along a selected axis (Sagittal, Coronal, or Axial).
It supports 3D or 4D medical image formats, allowing users to specify a slice range for more focused visualization.
The script also allows users to choose between a minimum or maximum projection based on their needs.

Arguments:
----------
in_image: str
    Path to the input NIfTI image.
axe: int (0: Sagittal, 1: Coronal, 2: Axial)
    The axis along which to project the image data.
--minmax: int (default: 0)
    0 for minimum intensity projection, 1 for maximum intensity projection.
--start: int (default: None)
    Starting slice index for the projection.
--end: int (default: None)
    Ending slice index for the projection.
"""

import argparse
import nibabel as nib
import tools.math
import tools.io
import tools.display
import matplotlib.pyplot as plt
import numpy as np


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)
    p.add_argument('in_image',
                   help='Input image.')
    p.add_argument('axe', type=int, default=0,
                        help='Axe de la vue (Sagittale 0, Coronale 1, Axiale 2)')
    p.add_argument('--minmax', type=int, default=0,
                        help='Min projection 0, Max projection 1')
    p.add_argument('--start', type=int, default=None,
                        help='Starting slice index for projection')
    p.add_argument('--end', type=int, default=None,
                        help='Ending slice index for projection')
    

    return p

def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    in_image = args.in_image
    axe = args.axe
    start_idx = args.start
    end_idx = args.end
    
    # Check if the input image is valid
    image = tools.io.check_valid_image(in_image)
    
    if image:
        # Validate the selected axis
        if axe not in (0, 1, 2):
            raise ValueError(f"Error: Invalid axis '{axe}' selected. Please choose one of the following:\n"
                             f" - Sagittal (0)\n"
                             f" - Coronal (1)\n"
                             f" - Axial (2)")
            
        # Get the slice range for the selected axis
        slice_range = image.shape[axe]
        
        # Set default values for slice range if not provided
        if start_idx is None:
            start_idx = 0
        if end_idx is None:
            end_idx = slice_range - 1
    
        # Validate slice range
        if start_idx > end_idx:
            raise ValueError("Error: Start index must be smaller than or equal to end index for slice range")
        if start_idx < 0 or end_idx < 0 or start_idx >= slice_range or end_idx >= slice_range:
            raise ValueError(f"Error: Indices must be within the valid slice range (0 to {slice_range - 1})")
        
        # Reorient image data if needed
        data = tools.io.reorient_data_rsa(image)
        voxel_sizes = image.header.get_zooms()
        
        # Calculate Min/Max Intensity Projection
        projected_data = tools.math.calc_projection(data, axe, args.minmax, start_idx, end_idx)
        
        # Display the projected data
        tools.display.display_image(projected_data, voxel_sizes, axe)

if __name__ == "__main__":
    main()