#!/usr/bin/env python

"""
Module Description:
-------------------
This script visualizes the q-space sampling pattern in a 3D interactive plot. 

Usage:
------
show_q_space <path_to_gradient_table>

Parameters:
-----------
- g_tab: Path to the gradient table file. The file should contain b-values 
         and gradient directions, with one direction per row and the b-value 
         as the last column. The first row is ignored.

Output:
-------
- A 3D interactive visualization of the q-space sampling pattern.
"""

import argparse
from tools.display import display_q_space
from tools.io import read_grad_file

def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)
    # Add arguments here
    p.add_argument('g_tab', type=str, help='Path to the gradient table.')
    return p

def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    g_tab = args.g_tab

    bval, bvecs = read_grad_file(g_tab)
    display_q_space(bval,bvecs)


if __name__ == "__main__":
    main()
