#! /usr/bin/env python

import argparse
from tools import io, display

def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)

    p.add_argument('in_image_1',
                   help='Input image 1.'),
    p.add_argument('in_image_2',
                   help='Input image 2.')
    p.add_argument('--bins',
                   help='number of bins.')
    return p

#TODO adjust function witj args.bins
def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    
    bins = 256

    data1 = io.image_to_data(args.in_image_1)
    data2 = io.image_to_data(args.in_image_2)

    if data1.min() >= 0 and data1.max() <= 1:
        print("Rescaling and discretizing image 1 ...")
        data1=io.rescale_and_discretize_image(data1, 255)
    if data2.min() >= 0 and data2.max() <= 1:
        print("Rescaling and discretizing image 2 ...")
        data2=io.rescale_and_discretize_image(data2, 255)

    display.display_joint_hist(data1,data2)

if __name__ == "__main__":
    main()
