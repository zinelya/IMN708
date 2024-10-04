import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

def update_display(axe, data, current_slice, axe_view, voxel_sizes, is_4d, current_time):
    """
    Update the displayed image on the given axis based on current slice and time point.

    Parameters
    ----------
    axe: Matplotlib Axes object
        Matplotlib axis to update.
    data: numpy array.
        Numpy array of the image to view.
    current_slice: 
        Current slice to display.
    axe_view: 0 or 1 or 2
        axis for slice selection (0: Sagittal, 1: Coronal, 2: axial).
    voxel_sizes: tuple
        Spatial resolution.
    is_4d:  Bool
        Boolean flag indicating if the data is 4D.
    current_time: 
        Current time point (for 4D data).
    """
    axe.clear()  # Clear the previous image

    # Slice the data based on the selected axis and time point
    if axe_view == 0:  # Sagittal
        img_slice = data[current_slice, :, :, current_time] if is_4d else data[current_slice, :, :]
    elif axe_view == 1:  # Coronal
        img_slice = data[:, current_slice, :, current_time] if is_4d else data[:, current_slice, :]
    elif axe_view == 2:  # Axial
        img_slice = data[:, :, current_slice, current_time] if is_4d else data[:, :, current_slice]

    # Transpose and flip if needed to maintain correct orientation
    transposed_slice = img_slice.T
    if axe_view > 0:
        transposed_slice = np.fliplr(transposed_slice)

    # Set aspect ratio based on voxel_sizesel sizes
    aspect_ratios = [(voxel_sizes[1] / voxel_sizes[2]),  # Sagittal 
                     (voxel_sizes[0] / voxel_sizes[2]),  # Coronal 
                     (voxel_sizes[0] / voxel_sizes[1])]  # Axial 

    # Define projection titles
    titles = ['Sagittal', 'Coronal', 'Axial']
    axe.imshow(transposed_slice, cmap='gray', origin='lower', aspect=aspect_ratios[axe_view])
    axe.set_title(f'{titles[axe_view]} slice {current_slice}')
    plt.draw()


def display_image(data, voxel_sizes, axe, titre='Image'):
    """
    Display the image in 3D slices, allowing for scrolling through slices and time points if 4D.

    Parameters
    ----------
    data: numpy array.
        Numpy array of the image to view.
    voxel_sizes: Tuple
        Spatial resolution of the image in terms of voxel dimensions.
    axe: (0: Sagittal, 1: Coronal, 2: Axial)
        Axis to display slices.
    titre: string
        Title for the plot.
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


def display_stats(data, bins, title='', min_range=None, max_range=None, taille=None, voxel_size=None, unit=None, size_4d=None,  mean=None, Michelson=None, RMS=None, std_bg=None, std=None, SNR=None):
    """
    Display the histogram and the statistics of the image data.

    Parameters
    ----------
    data : numpy array
        Numpy array of the image.
    bins : int
        Number of bins to be used for the histogram.
    title : str
        Title for the plot.
    min_range : int
        Minimum intensity value to be included in the histogram. 
    max_range : int
        Maximum intensity value to be included in the histogram.
    taille : tuple of floats
        Dimensions of the image in the format (width, height, depth). 
    voxel_size : tuple of floats
        Spatial resolution of the image in terms of voxel dimensions.
    unit : str
        Unit of measurement for the voxel size or image dimensions, e.g., "mm".
    size_4d : int
        Size of the 4th dimension
    Michelson : float
        Michelson contrast value of the image.
    RMS : float
        Root mean square (RMS) contrast of the image.
    std_bg : float
        Standard deviation of the background noise in the image.
    std : float
        Standard deviation.
    SNR : float
        Signal-to-noise ratio (SNR) of the image.
    """
    # Apply the range filtering if min_range and max_range are provided
    if min_range is not None:
        data = data[data >= min_range]  # Keep data above or equal to min_range
    if max_range is not None:
        data = data[data <= max_range]
    
    # Create the histogram
    n, bins, patches = plt.hist(data.flatten(), bins=bins, density=True, edgecolor='black')
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    # Collect the statistics in a string
    stats_text = ""
    if taille is not None:
        stats_text += f"Resolution:  {voxel_size[0]:.3f}{unit[0]}x {voxel_size[1]:.3f}{unit[0]}x {voxel_size[2]:.3f}{unit[0]}\n"
    if voxel_size is not None:
        stats_text += f"Taille de l'image: {taille[0]} x {taille[1]} x {taille[2]}\n" 
    if size_4d is not None:
        stats_text += f"Taille de la 4eme dimension: {size_4d}\n"
    if Michelson is not None:
        stats_text += f"Michelson: {Michelson}\n"
    if RMS is not None:
        stats_text += f"RMS: {RMS:.3f}\n"
    if std_bg is not None:
        stats_text += f"Ecart-type du background: {std_bg:.2f}\n"
    if std is not None:
        stats_text += f"Ecart-type de la ROI: {std:.2f}\n"
    if SNR is not None:
        stats_text += f"SNR: {SNR:.3f}\n"

    plt.gcf().text(0.72, 0.5, stats_text, fontsize=12, bbox=dict(facecolor='blue', alpha=0.5))
    plt.subplots_adjust(right=0.7)
    plt.show()
