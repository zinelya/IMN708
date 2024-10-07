# TP1_IMN708

Welcome to **TP1** of the [IMN530/708 Course](https://scil.usherbrooke.ca/courses/imn530/)! 


## Installation

To get started with **TP1_IMN708**, you should install it in a virtual environment.

### Follow these steps:

1. **Clone this repository** to your local machine:
    ```bash
    git clone https://github.com/yourusername/TP1_IMN708.git
    cd TP1_IMN708
    ```

2. **Create a virtual environment** (or activate it if already created).

3. **Install the library**:
    Once your virtual environment is activated, run the following command to install the project:
    ```bash
    pip install -e .
    ```
## Usage

This installation will provide you with access to the following useful commands:

- `view_image` 📷 — For viewing the NIfTI images.
    ```bash
    view_image <image.nii> <axis : Sagittale 0, Coronale 1, Axiale 2> [—-title <Title of the image>]
    ```
    
- `compute_mIP_MIP` 🔍 — For performing minimum/Maximum Intensity Projection.
  ```bash 
  compute_mip_MIP <image.nii> <axis: Sagittale 0, Coronale 1, Axiale 2> [--minmax <projection: min_intensity 0, max_intensity 1>] [--start <starting_slice_index>] [--end <ending_slice_index>]
   ```
  
- `stats` 📊 — To compute and display statistics of the image.
    ```bash 
    stats <image.nii> [--labels <label_image.nii>] [--bins <number_of_bins>] [--min_range <min_value>] [--max_range <max_value>]
    ```
    
- `denoise` 🧹 — To apply denoising filters to images.
    ```bash
   denoise <in_image.nii> <denoise_method: 0 NLMs, 1 Gaussian, 3 Median, 4 Bilateral, 5 Anisotropic Diffusion> [--output_dir <directory>] [--axe <axis to visualize denoised image : Sagittale 0, Coronale 1, Axiale 2>] [--sigma <value>] [--patch_size <size>] [--patch_distance <distance>] [--h <value>] [--sigma_color <value>] [--sigma_spatial <value>] [--n <value>] [--kappa <value>] [--gamma <value>] 
    ```
All commands come with a helpful `--help` parameter, so you can easily learn how to use each one. Just run:

```bash
<command> --help
```
    
## The Data

We have transformed all the data to NIfTI format for generalized usage.

ROIs were also hand delineated from the given data.

You can find the data folder in the repository, except for the rat111.nii, than you can download from this [link](https://usherbrooke-my.sharepoint.com/:u:/g/personal/nguc4116_usherbrooke_ca/EY_bqpQ-jGBOkGopPhykAiwBptfaZ8PevcJeYeg6sWPqEQ?xsdata=MDV8MDJ8WmluZWIuRWwuWWFtYW5pQFVTaGVyYnJvb2tlLmNhfDc3Mzg0MjA5YWEwNzRjNGIzODZhMDhkY2U2NjkwNzI3fDNhNWE4NzQ0NTkzNTQ1Zjk5NDIzYjMyYzNhNWRlMDgyfDB8MHw2Mzg2Mzg1ODY0ODg5NjE1Njl8VW5rbm93bnxUV0ZwYkdac2IzZDhleUpXSWpvaU1DNHdMakF3TURBaUxDSlFJam9pVjJsdU16SWlMQ0pCVGlJNklrMWhhV3dpTENKWFZDSTZNbjA9fDB8fHw%3d&sdata=ODlDSE5KNGVibDlsU3lBU0Jud0k2VEhVZ3R5YzJzdUU4SzJzYzJKbDNCZz0%3d) and add to the data folder.

