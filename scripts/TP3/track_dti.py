#!/usr/bin/env python

"""
Module Description:
-------------------
This script performs deterministic tractography using diffusion MRI data. It generates streamlines from the provided
diffusion peaks and masks and saves the results in TRK format.

Features:
---------
- Load diffusion peaks and generate stop and seed masks.
- Perform deterministic tracking with adjustable parameters.
- Save the generated streamlines as a TRK file.

Usage:
------
Example of running the script:
    ./tractography.py peaks.nii.gz fa.nii.gz dwi.nii.gz output.trk

Parameters:
-----------
- `input_peaks`: Path to the input peaks file (4D NIfTI format).
- `fa`: Path to the fractional anisotropy (FA) map (3D NIfTI format).
- `dwi`: Path to the DWI reference image (3D NIfTI format).
- `output`: Path to save the generated tracts (TRK format).
"""

import argparse
import numpy as np
import nibabel as nib
from dipy.io.streamline import save_trk
from dipy.io.stateful_tractogram import StatefulTractogram, Space
from tools.tracking import track_streamlines  # Assuming your tracking logic is in tools.tracking

def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)
    p.add_argument('input_peaks', help='Path to the input peaks file (4D NIfTI format).')
    p.add_argument('fa', help='Path to the fractional anisotropy (FA) map (3D NIfTI format).')
    p.add_argument('dwi', help='Path to the DWI reference image (4D NIfTI format).')
    p.add_argument('output', help='Path to save the generated tracts (TRK format).')
    p.add_argument('--seed_density', type=int, default=10, help='Number of seeds within each voxel (default: 10).')
    p.add_argument('--step_size', type=float, default=1.0, help='Step size for streamline advancement (default: 1.0).')
    p.add_argument('--angle_threshold', type=float, default=40.0, help='Angle threshold in degrees (default: 40).')
    p.add_argument('--min_length', type=float, default=10.0, help='Minimum length of a streamline in mm (default: 5.0).')
    p.add_argument('--max_length', type=float, default=100.0, help='Maximum length of a streamline in mm (default: 100.0).')
    return p

def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Load input data
    print("Loading data...")
    peaks_img = nib.load(args.input_peaks)
    peaks = peaks_img.get_fdata()

    fa_img = nib.load(args.fa)
    fa = fa_img.get_fdata()

    dwi_img = nib.load(args.dwi)

    # Generate masks
    stop_mask = (fa >= 0.15)
    seed_mask = np.copy(stop_mask)

    # Perform deterministic tracking
    print("Performing deterministic tracking...")
    tracts = track_streamlines(
        peaks=peaks,
        stop_mask=stop_mask,
        seed_mask=seed_mask,
        seed_density=args.seed_density,
        step_size=args.step_size,
        angle_threshold=args.angle_threshold,
        min_length=args.min_length,
        max_length=args.max_length
    )
    print(f"Number of tracts generated: {len(tracts)}")

    # Create tractogram
    print("Creating tractogram...")
    tractogram = StatefulTractogram(tracts, dwi_img, Space.VOX)

    # Save tractogram
    print(f"Saving tractogram to {args.output}...")
    save_trk(tractogram, args.output)

    print("Tractography completed successfully.")

if __name__ == "__main__":
    main()