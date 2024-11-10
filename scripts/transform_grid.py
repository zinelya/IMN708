"""
This script reads 
"""

import argparse
import tools.display
import tools.math
import tools.transform
import numpy as np


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)

    p.add_argument('--w', type=int, default=5,
                   help='Width of the 3D grid (x-axis)')
    p.add_argument('--d', type=int, default=5,
                   help='Depth of the 3D grid (y-axis)')
    p.add_argument('--h', type=int, default=5,
                   help='Height of the 3D grid (z-axis)')
    p.add_argument('--x', type=float, default=0,
                   help='Initial x-axis position of the grid')
    p.add_argument('--y', type=float, default=0,
                   help='Initial y-axis position of the grid')
    p.add_argument('--z', type=float, default=0,
                   help='Initial z-axis position of the grid')
    p.add_argument('--p', type=float, default=0,
                   help='Translation distance in x-axis')
    p.add_argument('--q', type=float, default=0,
                   help='Translation distance in y-axis')
    p.add_argument('--r', type=float, default=0,
                   help='Translation distance in z-axis')
    p.add_argument('--theta', type=float, default=0,
                   help='Rotation angle along the x-axis')
    p.add_argument('--omega', type=float, default=0,
                   help='Rotation angle along the y-axis')
    p.add_argument('--phi', type=float, default=0,
                   help='Rotation angle along the z-axis')
    p.add_argument('--s', type=float, default=1,
                   help='Scaling factor')
    p.add_argument('--affine', type=str, help='Path to the affine transformation matrix file')
    
    return p


def load_affine_matrix(file_path):
    """
    Load affine transformation matrix from a text file.
    """
    with open(file_path, 'r') as f:
        matrix = np.loadtxt(f)
    return matrix

def main():
    # Parse the arguments
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Initialize N*3 array for points in the 3D grid
    grid = tools.math.initialize_grid_points(args.w, args.d, args.h, args.x, args.y, args.z)

    # Apply rigid transformation if rotation/translation parameters are provided
    transformed_grid = tools.transform.trans_rigide(grid, args.theta, args.omega, args.phi, args.p, args.q, args.r)

    # Apply scaling if specified
    scaled_grid = tools.transform.similitude(transformed_grid, args.s)

    if args.affine:
        affine = load_affine_matrix(args.affine)
        # Apply the affine transformation
        transformed_grid = tools.transform.trans_with_affine_matrix(scaled_grid, affine)
        # Displays the grids before an after transformation and prints SVD results
        tools.display.display_grids([grid, transformed_grid])
    else:
        tools.display.display_grids([grid, scaled_grid])
    
            
if __name__ == "__main__":
    main()