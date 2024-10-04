#! /usr/bin/env python

"""
Description of what the script does
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
    

    return p

def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    in_image = args.in_image
    axe = args.axe
    
    image = tools.io.check_valid_image(in_image)
    
    if image:
        if axe not in (0,1,2):
            raise ValueError(f"Error: Invalid axis '{axe}' selected. Please choose one of the following:\n"
                         f" - Sagittal (0)\n"
                         f" - Coronal (1)\n"
                         f" - Axial (2)")
        data = tools.io.reorient_data_rsa(image)
        voxel_sizes = image.header.get_zooms()
        
        projected_data = tools.math.calc_projection(data, axe, args.minmax)
        tools.display.display_image(projected_data, voxel_sizes, axe)


if __name__ == "__main__":
    main()
