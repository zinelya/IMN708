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

This installation will provide you with access to the following useful commands:

- `view_image` 📷 — For viewing the NIfTI images.
- `mIP` 🔍 — For performing minimum Intensity Projection.
- `MIP` 🏞️ — To display the Maximum Intensity Projection.
- `stats` 📊 — To compute and display statistics of the image.
- `denoise` 🧹 — To apply denoising filters to images.

## Getting the Data

We have transformed all the data to NIfTI format for generalized usage.

### To get the data:
1. Ensure you are in the `TP1_IMN708` folder.
2. Run the following command to extract the data:
```bash
unzip data.zip
```

## Usage

All commands come with a helpful `--help` parameter, so you can easily learn how to use each one. Just run:

```bash
<command> --help
```
