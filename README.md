<div align="center">

🌈 Welcome to **TP1** and **TP2** of the [IMN530/708 Course](https://scil.usherbrooke.ca/courses/imn530/)! 🌈

</div>

## Installation

To get started with this project, you should install it in a virtual environment.

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
We are actively working on upgrading this project, so please consider pulling the main branch every once in a while ! 😃

# TP2 : Registration

This installation will provide you with access to the following useful commands:

- `compare_pairs` 🕵🏻 — For comparing two 2D images.
    ```bash
    view_image <in_image_1> <in_image_2> [—-bins <number of bins>]
    ```
    
- `transform_grid` 🔄 — To apply transformations and visualize grids.
  ```bash 
  transform_grid.py [--w Width of grid (along x-axis)] [--d depth of grid (along y-axis)] [--h height of grid (along z-axis)] [--x initial x-axis position] [--y initial y-axis position] [--z initial z-axis position] [--p translation distance in x-axis] [--q translation distance in y-axis] [--r translation distance in y_axis] [--theta rotation angle in degree along the x-axis] [--omega rotation angle along the y_axis] [--phi rotation angle along the z_axis] [--s scaling factor] [--affine Path to affine transformation matrix file]
   ```
  
- `register_image` 📏 — For registering an image to a reference.
    ```bash 
    register_image.py [fix_image Path to fixed image] [transform_method Transformation method: 0=Translation, 1=Rotation, 2=Rigid, 3=Translation-based registration, 4=Rotation-based registration, 5=Rigid registration] [--mov_image Path to moving image] [--p Translation distance along x-axis] [--q Translation distance along y-axis] [--theta Rotation angle in degrees] [--out_dir Directory to save transformed image] [--eta_t Learning rate for translation] [--eta_r Learning rate for rotation] [--n_iter Maximum number of iterations for stopping] [--conv_thresh SSD convergence threshold] [--max_std Maximum standard deviation for SSD] [--n_last Number of last iterations for SSD stability] [--res_level Resolution hierarchy level] [--gauss_sigma Sigma for Gaussian image denoising in multi-resolution method] [--grad_optimizer Gradient descent optimizer for rigid registration: 0=Regular, 1=Momentum-based] [--momentum Momentum factor]
    ```

All commands come with a helpful `--help` parameter, so you can easily learn how to use each one. Just run:

```bash
<command> --help
```
## data_TP2

This folder includes:

- **I and J images** — To compare and undertsand similiraty analysis.
- **Brains**  — For playing around with image registration and transformation exercises.
- **Affines**  — Added to visualize transformations for question 3(c).

To extract the data, start a terminal from the project folder and run the following commands:

To extract the data, open a terminal from the project folder and run the following commands:

```bash
unzip data_TP2.zip
rm data_TP2.zip
```

# TP1 : Medical Imaging Modality, Noise, and Denoising

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
    
## data_TP1

All the data have been converted to NIfTI format for a generalized usage. 

Additionally, Regions of Interest (ROIs) were manually delineated based on the provided data.

The zipped data folder is available within the repository, except for the file `rat111.nii`, which can be downloaded separately from this [link](https://usherbrooke-my.sharepoint.com/:u:/g/personal/nguc4116_usherbrooke_ca/EY_bqpQ-jGBOkGopPhykAiwBptfaZ8PevcJeYeg6sWPqEQ?xsdata=MDV8MDJ8WmluZWIuRWwuWWFtYW5pQFVTaGVyYnJvb2tlLmNhfDc3Mzg0MjA5YWEwNzRjNGIzODZhMDhkY2U2NjkwNzI3fDNhNWE4NzQ0NTkzNTQ1Zjk5NDIzYjMyYzNhNWRlMDgyfDB8MHw2Mzg2Mzg1ODY0ODg5NjE1Njl8VW5rbm93bnxUV0ZwYkdac2IzZDhleUpXSWpvaU1DNHdMakF3TURBaUxDSlFJam9pVjJsdU16SWlMQ0pCVGlJNklrMWhhV3dpTENKWFZDSTZNbjA9fDB8fHw%3d&sdata=ODlDSE5KNGVibDlsU3lBU0Jud0k2VEhVZ3R5YzJzdUU4SzJzYzJKbDNCZz0%3d).

To extract the data, start a terminal from the project folder and run the following commands:

```bash
unzip data_TP1.zip
rm data_TP1.zip
```
