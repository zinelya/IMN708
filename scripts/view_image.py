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
    p.add_argument('--axe', type=int, default=0,
                        help='Axe de la vue (Sagittale 0, Coronale 1, Axiale 2)')
    
    tools.utils.add_verbose_arg(p)

    return p

def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Read arguments
    in_image = args.in_image
    axe = args.axe
    
    # Call view function
    tools.image.afficher_image(in_image, axe)
        

if __name__ == "__main__":
    main()
