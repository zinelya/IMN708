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

def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)
    # Add arguments here
    p.add_argument('input', help='Path to the input file.')
    p.add_argument('--optional_arg', type=str, help='Optional argument example.')
    return p

def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

if __name__ == "__main__":
    main()

