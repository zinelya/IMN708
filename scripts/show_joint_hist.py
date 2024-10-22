#! /usr/bin/env python
"""
#TODO
This script displays intensity histograms and statistics
of the entire NIfTI image or specific regions of interest (RoIs).

Features:
- Display an intensity histogram of the input image with configurable bins, range.
- Support for both 2D, 3D and 4D images.
- Optionally, display histograms for specific labels (regions of interest) defined in a separate label mask. (e.g t1_labels.nii.gz for t1.nii.gz)
- Calculate image quality metrics such as standard deviation of the background, mean intensity for each RoI, 
  and SNR (Signal-to-Noise Ratio).

Usage:
------
Example of running the script:

    stats <image> --labels <label_image> --bins 100 --min_range 10 --max_range 500 

Parameters:
-----------
in_image: str
    Path to the input image.
--labels: str, optional
    Path to a label image, used to display histograms for each region of interest (RoI). 
    If provided, the histogram will be computed for each unique label in the image.
--bins: int, optional
    Number of bins to use for the intensity histogram. Default is 100.
--min_range: float, optional
    Minimum intensity value for the histogram range.
--max_range: float, optional
    Maximum intensity value for the histogram range.
"""


import argparse
from tools import io, display

def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)

    p.add_argument('in_image_1',
                   help='Input image 1.'),
    p.add_argument('in_image_2',
                   help='Input image 2.')
    p.add_argument('--bins', type=int, default=256,
                   help='number of bins.')
    return p

#TODO adjust function with args.bins
def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    

    data1 = io.image_to_data(args.in_image_1)
    data2 = io.image_to_data(args.in_image_2)

    if data1.min() >= 0 and data1.max() <= 1:
        print("Rescaling and discretizing image 1 ...")
        data1=io.rescale_and_discretize_image(data1, args.bins - 1)
    if data2.min() >= 0 and data2.max() <= 1:
        print("Rescaling and discretizing image 2 ...")
        data2=io.rescale_and_discretize_image(data2, args.bins - 1)

    display.display_joint_hist(data1,data2,bins=args.bins)

if __name__ == "__main__":
    main()
