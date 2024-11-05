#! /usr/bin/env python

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
    
    return p

def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Initialize N*3 array for points in 3d_grid
    grid = tools.transform.initialize_grid_points(args.w, args.d, args.h, args.x, args.y, args.z)
    # transformed_grid = tools.transform.trans_rigide(grid, args.theta, args.omega, args.phi, args.p, args.q, args.r)
    # scaled_grid = tools.transform.similitude(transformed_grid, args.s)
    # tools.display.display_grids([grid, scaled_grid])
    
    # # Test affine matrix multiplication
    # affine_matrix_1 = np.array([[0.9045, -0.3847, -0.1840, 10], [0.2939,0.875, -0.3847, 10], [0.3090, 0.2939, 0.9045, 10], [0, 0, 0, 1]])
    # affine_grid_1 = tools.transform.trans_with_affine_matrix(grid, affine_matrix_1)
    
    # affine_matrix_2 = np.array([[0, -0.2598, 0.15, -3], [0, -0.15, -0.2598, 1.5], [0.3, 0, 0, 0], [0, 0, 0, 1]])
    # affine_grid_2 = tools.transform.trans_with_affine_matrix(grid, affine_matrix_2)
    
    affine_matrix_3 = np.array([[0.7182, -1.3727, -0.5660, 1.8115], [-1.9236, -4.6556, -2.5512, 0.2873], [-0.6426, -1.7985, -1.6285, 0.7404], [0, 0, 0, 1]])
    affine_grid_3 = tools.transform.trans_with_affine_matrix(grid, affine_matrix_3)
    tools.display.display_grids([grid, affine_grid_3])
    return 
    
            
if __name__ == "__main__":
    main()