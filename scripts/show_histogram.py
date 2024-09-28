#! /usr/bin/env python

"""
Description of what the script does
"""

import argparse
import tools.utils
import tools.image


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)

    p.add_argument('in_image',
                   help='Input image.')
    p.add_argument('--bin', type=int, default=100,
                        help='Number of bins')
    p.add_argument('--in_mask', type=str, default='',
                        help='If in_mask is set, show histogram for each segment in the mask')
    
    p.add_argument('--min_range', type=float, default=None,
                   help='Minimum value for the histogram range (default: None).')

    p.add_argument('--max_range', type=float, default=None,
                   help='Maximum value for the histogram range (default: None).')
    
    tools.utils.add_verbose_arg(p)

    return p

def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Read arguments
    in_image = args.in_image
    bin = args.bin
    min_range = args.min_range
    max_range = args.max_range
    in_mask = args.in_mask
    
    # Call view function
    tools.image.plot_histogram(in_image, bin, min_range, max_range, in_mask)
        

if __name__ == "__main__":
    main()
