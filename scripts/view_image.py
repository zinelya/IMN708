#! /usr/bin/env python

"""
Description of what the script does
"""

import argparse
import tools.utils
<<<<<<< HEAD
import tools.image
=======

>>>>>>> 00a804f0f49e53ed5663e0b9e359b8151bb65107


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)

    p.add_argument('in_image',
                   help='Input image.')
<<<<<<< HEAD
    p.add_argument('--axe', type=int, default=0,
                        help='Axe de la vue (Sagittale 0, Coronale 1, Axiale 2)')
=======
    p.add_argument('--optional', default=0,
                   help='optional argument.')
>>>>>>> 00a804f0f49e53ed5663e0b9e359b8151bb65107
    
    tools.utils.add_verbose_arg(p)

    return p

def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

<<<<<<< HEAD
    # Read arguments
    in_image = args.in_image
    axe = args.axe
    
    # Call view function
    tools.image.afficher_image(in_image, axe)
        
=======
    #TODO replace the code here
    print(args.in_image)
    if args.optional:
        print(args.optional)
>>>>>>> 00a804f0f49e53ed5663e0b9e359b8151bb65107

if __name__ == "__main__":
    main()
