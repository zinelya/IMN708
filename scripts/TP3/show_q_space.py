#!/usr/bin/env python

"""
Module Description:
-------------------
[Provide a brief description of the functionality of the script.]

Features:
---------
- [Feature 1]
- [Feature 2]
- [Feature 3]

Usage:
------
Example of running the script:
    [Provide example command-line usage.]

Parameters:
-----------
[Provide details of each parameter and its purpose.]
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
