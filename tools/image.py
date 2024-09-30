import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
import tools.io

def reorient_data_rsa(image):
    """
    Reorient the image to 'RPS' orientation by flipping along the necessary axes.
    :param image: NIfTI image object.
    :return: Reoriented data array.
    """
    data = image.get_fdata()
    x, y, z = nib.aff2axcodes(image.affine)
    
    # Flip the image data along appropriate axes to achieve 'RPS' orientation.
    if x != 'R':
        data = np.flip(data, axis=0)
    if y != 'A':
        data = np.flip(data, axis=1)
    if z != 'S':
        data = np.flip(data, axis=2)

    return data

def update_display(ax, data, current_slice, axe, voxel_sizes, is_4d, current_time):
    """
    Update the displayed image on the given axis based on current slice and time point.
    :param ax: Matplotlib axis to update.
    :param data: Image data.
    :param current_slice: Current slice to display.
    :param axe: Axis for slice selection (0: Sagittal, 1: Coronal, 2: Axial).
    :param voxel_sizes: Tuple of voxel sizes for aspect ratio adjustments.
    :param is_4d: Boolean flag indicating if the data is 4D.
    :param current_time: Current time point (for 4D data).
    """
    ax.clear()  # Clear the previous image

    # Slice the data based on the selected axis and time point
    if axe == 0:  # Sagittal
        img_slice = data[current_slice, :, :, current_time] if is_4d else data[current_slice, :, :]
    elif axe == 1:  # Coronal
        img_slice = data[:, current_slice, :, current_time] if is_4d else data[:, current_slice, :]
    elif axe == 2:  # Axial
        img_slice = data[:, :, current_slice, current_time] if is_4d else data[:, :, current_slice]

    # Transpose and flip if needed to maintain correct orientation
    transposed_slice = img_slice.T
    if axe > 0:
        transposed_slice = np.fliplr(transposed_slice)

    # Set aspect ratio based on voxel sizes
    aspect_ratios = [(voxel_sizes[1] / voxel_sizes[2]),  # Sagittal (X-axis)
                     (voxel_sizes[0] / voxel_sizes[2]),  # Coronal (Y-axis)
                     (voxel_sizes[0] / voxel_sizes[1])]  # Axial (Z-axis)

    # Define projection titles
    titles = ['Sagittal', 'Coronal', 'Axial']
    ax.imshow(transposed_slice, cmap='gray', origin='lower', aspect=aspect_ratios[axe])
    ax.set_title(f'{titles[axe]} slice {current_slice}')
    plt.draw()

def display_image(data, voxel_sizes, axe, titre='Image'):
    """
    Display the image in 2D slices, allowing for scrolling through slices and time points if 4D.
    :param in_image: NIfTI image file.
    :param axe: Axis to display slices (0: Sagittal, 1: Coronal, 2: Axial).
    :param titre: Title for the plot.
    """
    is_4d = data.ndim == 4
    current_time = 0
    num_timepoints = data.shape[3] if is_4d else 1
    current_slice = data.shape[axe] // 2
    total_slices = data.shape[axe]
    
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)  # Space for time slider if needed

    def on_scroll(event):
        """Handle mouse scroll event to change the slice."""
        nonlocal current_slice
        if event.button == 'up' and current_slice < total_slices - 1:
            current_slice += 1
        elif event.button == 'down' and current_slice > 0:
            current_slice -= 1
        update_display(ax, data, current_slice, axe, voxel_sizes, is_4d, current_time)

    def on_time_slider(val):
        """Handle changes in time slider for 4D images."""
        nonlocal current_time
        current_time = int(val)
        update_display(ax, data, current_slice, axe, voxel_sizes, is_4d, current_time)

    # Connect the scroll event to the figure for slice navigation
    fig.canvas.mpl_connect('scroll_event', on_scroll)

    # Create a time slider if the image is 4D
    if is_4d and num_timepoints > 1:
        ax_time = plt.axes([0.25, 0.1, 0.65, 0.03])  # Position of the time slider
        time_slider = Slider(ax_time, 'Time', 0, num_timepoints - 1, valinit=0, valstep=1)
        time_slider.on_changed(on_time_slider)

    # Display the initial slice
    update_display(ax, data, current_slice, axe, voxel_sizes, is_4d, current_time)
    plt.show()