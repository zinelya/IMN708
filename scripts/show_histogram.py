import argparse
import tools.utils
import numpy as np
import nibabel as nib
import tools.math


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)

    p.add_argument('in_image',
                   help='Input image.')
    p.add_argument('--bin', type=int, default=100,
                        help='Number of bins')
    p.add_argument('--min_range', type=float, default=None,
                   help='Minimum value for the histogram range.')
    p.add_argument('--max_range', type=float, default=None,
                   help='Maximum value for the histogram range.')
    p.add_argument('--in_labels', type=str, default='',
                        help='If in_labels is set, show histogram for each label.')
    
    
    tools.utils.add_verbose_arg(p)

    return p

def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    image = nib.load(args.in_image)
    data = image.get_fdata()
    
    # Extract image data as a NumPy array and rescale if negative value exists (CT scan)
    data = image.get_fdata()
    if np.any(data < 0):
        data = data - np.min(data)
    
    if args.in_labels:
        labels = nib.load(args.in_labels)
        labels_data = labels.get_fdata()
        if image.shape != labels.shape:
            print("The shapes of the original image and labels are different.")
            return
        tools.math.histogram(data, args.bin, args.min_range, args.max_range, labels_data)
    
    else:
        tools.math.histogram(data, args.bin, args.min_range, args.max_range)
    



if __name__ == "__main__":
    main()
