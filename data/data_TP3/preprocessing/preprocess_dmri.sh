#!/bin/bash

# Set up the directory containing the data
# Replace the path below with the actual path to your data_TP3/ 
Directory=".."

# Define file paths
dmri="${Directory}/dmri.nii.gz"
bvec="${Directory}/dmri.bvec"
bvals="${Directory}/dmri.bval"

# Check if necessary files exist
if [[ ! -f $dmri || ! -f $bvec || ! -f $bvals ]]; then
  echo "Error: One or more input files are missing in $Directory."
  exit 1
fi

# Create a directory for intermediate files
IntermediateDir="${Directory}/intermediate_files"
mkdir -p "$IntermediateDir"

# Step 1: Denoise
echo "Starting Step 1: Denoising..."
dwidenoise "$dmri" "$IntermediateDir/denoised_dmri.nii.gz" -f
if [[ $? -ne 0 ]]; then
  echo "Error in dwidenoise step."
  exit 1
fi
echo "Step 1 completed."

# Step 2: Brain extraction and masking
echo "Starting Step 2: Brain extraction..."
scil_dwi_extract_b0.py "$IntermediateDir/denoised_dmri.nii.gz" "$bvals" "$bvec" "$IntermediateDir/denoised_b0.nii.gz"
bet "$IntermediateDir/denoised_b0.nii.gz" "$IntermediateDir/bet_denoised_b0.nii.gz" -m -R -f 0.16
if [[ $? -ne 0 ]]; then
  echo "Error in BET step."
  exit 1
fi
python "$Directory/preprocessing/apply_mask.py" "$IntermediateDir/denoised_dmri.nii.gz" "$IntermediateDir/bet_denoised_b0_mask.nii.gz" "$IntermediateDir/bet_denoised_dmri.nii.gz"
echo "Step 2 completed."

# Step 3: Eddy correction
echo "Starting Step 3: Eddy correction..."
printf "0 -1 0 0.0646\n0 1 0 0.0646" > "$IntermediateDir/acqparams.txt"
indx=""
for ((i=1; i<=65; i+=1)); do indx="$indx 1"; done
echo "$indx" > "$IntermediateDir/index.txt"
eddy --imain="$IntermediateDir/bet_denoised_dmri.nii.gz" \
     --mask="$IntermediateDir/bet_denoised_b0_mask.nii.gz" \
     --acqp="$IntermediateDir/acqparams.txt" \
     --index="$IntermediateDir/index.txt" \
     --bvecs="$bvec" --bvals="$bvals" \
     --out="$IntermediateDir/eddy_bet_denoised_dmri"

if [[ $? -ne 0 ]]; then
  echo "Error in eddy step."
  exit 1
fi

# Ensure all values are positive
fslmaths "$IntermediateDir/eddy_bet_denoised_dmri.nii.gz" -thr 0.0001 "$IntermediateDir/pos_eddy_bet_denoised_dmri.nii.gz"
fslstats "$IntermediateDir/pos_eddy_bet_denoised_dmri.nii.gz" -R
echo "Step 3 completed."

# Step 4: Refining brain extraction
echo "Starting Step 4: Refining brain extraction..."
scil_dwi_extract_b0.py "$IntermediateDir/pos_eddy_bet_denoised_dmri.nii.gz" "$bvals" "$bvec" "$IntermediateDir/pos_eddy_bet_denoised_b0.nii.gz"
bet "$IntermediateDir/pos_eddy_bet_denoised_b0.nii.gz" "$IntermediateDir/bet_pos_eddy_bet_denoised_b0.nii.gz" -m -R -f 0.16
if [[ $? -ne 0 ]]; then
  echo "Error in BET step."
  exit 1
fi
python "$Directory/preprocessing/apply_mask.py" "$IntermediateDir/pos_eddy_bet_denoised_dmri.nii.gz" "$IntermediateDir/bet_pos_eddy_bet_denoised_b0_mask.nii.gz" "$IntermediateDir/bet_pos_eddy_bet_denoised_dmrii.nii.gz"
echo "Step 4 completed."

# Step 5: Bias field correction
echo "Starting Step 5: Bias field correction..."
N4BiasFieldCorrection -i "$IntermediateDir/bet_pos_eddy_bet_denoised_b0.nii.gz" \
                      -o ["$IntermediateDir/n4_bet_pos_eddy_bet_denoised_b0.nii.gz", "$IntermediateDir/bias_field.nii.gz"] \
                      -c [300x150x75x50, 1e-6] -v 1

if [[ $? -ne 0 ]]; then
  echo "Error in N4BiasFieldCorrection step."
  exit 1
fi

scil_dwi_apply_bias_field.py "$IntermediateDir/bet_pos_eddy_bet_denoised_dmrii.nii.gz" "$IntermediateDir/bias_field.nii.gz" "$IntermediateDir/bias_corrected_dmri.nii.gz" --mask "$IntermediateDir/bet_pos_eddy_bet_denoised_b0_mask.nii.gz" -f
echo "Step 5 completed."

# Step 6: Upsampling to 1x1x1 mm resolution
echo "Starting Step 6: Upsampling to 1x1x1 mm resolution..."
mrgrid "$IntermediateDir/bias_corrected_dmri.nii.gz" regrid "$IntermediateDir/dmri_1x1x1.nii.gz" -voxel 1,1,1 -force
if [[ $? -ne 0 ]]; then
  echo "Error during upsampling step."
  exit 1
fi

# Step 7: Final brain extraction
echo "Starting Step 7: Final brain extraction..."
scil_dwi_extract_b0.py "$IntermediateDir/dmri_1x1x1.nii.gz" "$bvals" "$bvec" "$IntermediateDir/b0_1x1x1.nii.gz"
bet "$IntermediateDir/b0_1x1x1.nii.gz" "$IntermediateDir/bet_b0_1x1x1.nii.gz" -m -R -f 0.16
if [[ $? -ne 0 ]]; then
  echo "Error in BET step."
  exit 1
fi
python "$Directory/preprocessing/apply_mask.py" "$IntermediateDir/dmri_1x1x1.nii.gz" "$IntermediateDir/bet_b0_1x1x1_mask.nii.gz" "$Directory/ready_dmri_1x1x1.nii.gz"
echo "Step 7 completed."

# Clean up intermediate files if desired
echo "Cleaning up intermediate files..."
# Uncomment the following line to remove intermediate files after processing
# rm -r "$IntermediateDir"

echo "Processing complete. Final file: $Directory/ready_dmri_1x1x1.nii.gz"
