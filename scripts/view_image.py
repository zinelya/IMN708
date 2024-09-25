#! /usr/bin/env python

"""
Description of what the script does
"""

import argparse
import tools.utils



def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)

    p.add_argument('in_image',
                   help='Input image.')
    p.add_argument('--optional', default=0,
                   help='optional argument.')
    
    tools.utils.add_verbose_arg(p)

    return p

def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    #TODO replace the code here
    print(args.in_image)
    if args.optional:
        print(args.optional)

if __name__ == "__main__":
    main()
