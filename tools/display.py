import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
import plotly.graph_objs as go
import plotly.io as pio
from skimage.transform import resize


def update_display(axe, data, current_slice, axe_view, voxel_sizes, is_4d, current_time, title=''):
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
    title: string
        Title for the plot.
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

    # Set aspect ratio (height/width) based on voxel_sizesel sizes
    aspect_ratios = [(voxel_sizes[2] / voxel_sizes[1]),  # Sagittal 
                     (voxel_sizes[2] / voxel_sizes[0]),  # Coronal 
                     (voxel_sizes[1] / voxel_sizes[0])]  # Axial 

    # Define projection titles
    titles = ['Sagittal', 'Coronal', 'Axial']
    axe.imshow(transposed_slice, cmap='gray', origin='lower', aspect=aspect_ratios[axe_view])
    
    # If title is not set, title is defined by the slice index and view axis
    if title == '':
        title = f'{titles[axe_view]} slice {current_slice}'
    axe.set_title(title)
    plt.draw()


def display_image(data, voxel_sizes, axe, title=''):
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
    title: string
        Title for the plot.
    """
    is_4d = data.ndim == 4
    is_2d = data.ndim ==2
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
        update_display(ax, data, current_slice, axe, voxel_sizes, is_4d, current_time, title)

    # Connect the scroll event to the figure for slice navigation
    fig.canvas.mpl_connect('scroll_event', on_scroll)

    # Create a time slider if the image is 4D
    if is_4d and num_timepoints > 1:
        ax_time = plt.axes([0.25, 0.1, 0.65, 0.03])  # Position of the time slider
        time_slider = Slider(ax_time, 'Time', 0, num_timepoints - 1, valinit=0, valstep=1)
        time_slider.on_changed(on_time_slider)

    # Display the initial slice
    update_display(ax, data, current_slice, axe, voxel_sizes, is_4d, current_time, title)
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
        Size of the 4th dimension.
    mean : float
        Mean intensity
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
    n, bins, patches = plt.hist(data.flatten(), bins=bins, density=True, edgecolor='black', )
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    # Collect the statistics in a string
    stats_text = ""
    if taille is not None:
        stats_text += f"Resolution:  {voxel_size[0]:.3f}{unit[0]}x {voxel_size[1]:.3f}{unit[0]}x {voxel_size[2]:.3f}{unit[0]}\n"
    if voxel_size is not None:
        stats_text += f"Taille de l'image: {taille[0]} x {taille[1]} x {taille[2]} (voxel)\n" 
    if size_4d is not None:
        stats_text += f"Taille de la 4eme dimension: {size_4d}\n"
    if Michelson is not None:
        stats_text += f"Michelson: {Michelson}\n"
    if RMS is not None:
        stats_text += f"RMS: {RMS:.7f}\n"
    if mean is not None:
        stats_text += f"Intensite moyenne: {mean:.7f}\n"
    if std_bg is not None:
        stats_text += f"Ecart-type du background: {std_bg:.7f}\n"
    if std is not None:
        stats_text += f"Ecart-type de la ROI: {std:.7f}\n"
    if SNR is not None:
        stats_text += f"SNR: {SNR:.3f}\n"

    plt.gcf().text(0.72, 0.5, stats_text, fontsize=12, bbox=dict(facecolor='blue', alpha=0.5))
    plt.subplots_adjust(right=0.7)
    plt.show()

def display_joint_hist(joint_hist, min_i, max_i, min_j, max_j, bins):
    joint_hist = np.log(joint_hist+ 1e-6)
    joint_hist = joint_hist.T

    # Set up the plot
    plt.figure(figsize=(8, 6))
    plt.imshow(joint_hist, cmap='viridis', aspect='auto', extent=[min_i, max_i, min_j, max_j], origin='lower')

    # Add a color bar
    plt.colorbar(label='Log Frequency')

    # Label the axes
    plt.xlabel('Image I')
    plt.ylabel('Image J')
    
    # step_i = (max_i - min_i)/10
    # step_j = (max_j - min_j)/10
    
    # x_interval = [int(min_i + step_i*i) for i in range(0, 10)]
    # y_interval = [int(min_j + step_j*j) for j in range(0, 10)]
    # plt.xticks(ticks=x_interval, labels=x_interval)
    # plt.yticks(ticks=y_interval, labels=y_interval)

    plt.title('Joint Histogram of image I and J')

    # Show the plot
    plt.show()
    
def display_grids(grids):
    """
    Visualizes multiple 3D grids with distinct colors for each grid.
    
    Args:
        grids: List of 2D numpy arrays, where each array is of shape N*3 representing the points in the grid.
    """
    fig = go.Figure()

    # Define a list of colors for different grids (can be extended if more grids)
    colors = ['black', 'red', 'green', 'blue']
    
    for idx, grid_points in enumerate(grids):
        # Extract x, y, z coordinates from the grid
        x = grid_points[:, 0]
        y = grid_points[:, 1]
        z = grid_points[:, 2] if grid_points.ndim == 3 else np.zeros(grid_points.shape[0])
        
        # Create a scatter plot for each grid, with a distinct color
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=5,  # Size of the points
                color=colors[idx % len(colors)],  # Cycle through colors for each grid
                opacity=0.8,
            ),
            name=f'Grid {idx+1}'
        ))

    # Set axis labels and layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X-axis'),
            yaxis=dict(title='Y-axis'),
            zaxis=dict(title='Z-axis'),
        ),
        title="3D Grid Visualization",
        showlegend=True
    )

    # Show the figure
    pio.show(fig)

def resize_image(image_1, image_2):
    image_1_shape = image_1.shape  # Get the shape of image_1
    image_2_rescaled = resize(image_2, image_1_shape, anti_aliasing=True)
    
    return image_2_rescaled


def display_registration(fix_image, registered_images, ssd_arr, gradient_optimizer):
    num_images = len(registered_images)
    
    # Create the figure and layout with two subplots
    fig, (ax_img, ax_ssd) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Display the first image initially
    initial_idx = num_images - 1
    ax_img.imshow(resize_image(fix_image, registered_images[initial_idx]), cmap='gray')  # Adjust vmin/vmax as necessary
    ax_img.imshow(fix_image, cmap='Reds', alpha=0.2)  # Overlay fixed image with opacity 0.4
    ax_img.set_title(f'Loop 0, SSD: {ssd_arr[initial_idx]:.4f}')
    ax_img.axis('off')

    gradient_method = ''
    if gradient_optimizer == 0:
        gradient_method = 'Fix step'
    elif gradient_optimizer == 1:
        gradient_method = 'Regular (learning-rate)'
    elif gradient_optimizer == 2:
        gradient_method = 'Momentum'
    elif gradient_optimizer == 3:
        gradient_method = 'NAG (improved momentum)'
    # Plot the SSD array as a line graph
    
    ax_ssd.plot(ssd_arr, label='SSD over time')
    ssd_marker, = ax_ssd.plot(initial_idx, ssd_arr[initial_idx], 'ro')  # Initial marker for current loop
    ax_ssd.set_title(f'SSD Values (Loss function) - {gradient_method}')
    ax_ssd.set_xlabel('Loop')
    ax_ssd.set_ylabel('SSD')
    ax_ssd.legend()

    # Add a slider for adjusting the time (loop number)
    ax_slider = plt.axes([0.05, 0.005, 0.50, 0.03], facecolor='lightgray')  # Adjusted position
    slider = Slider(ax_slider, 'Time', 0, num_images - 1, valinit=initial_idx, valstep=1)

    # Function to update the image and SSD marker when slider is adjusted
    def update(val):
        ax_img.clear()
        loop_num = int(slider.val)
        moving_image = registered_images[loop_num]
        ax_img.imshow(resize_image(fix_image, moving_image), cmap='gray') 
        ax_img.imshow(fix_image, cmap='Reds', alpha=0.2)  # Overlay fixed image with opacity 0.4
        ax_img.set_title(f'Loop {loop_num}, SSD: {ssd_arr[loop_num]:.2f}, Size: {moving_image.shape[0]}x{moving_image.shape[1]}')
        ssd_marker.set_data(loop_num, ssd_arr[loop_num])  # Update SSD marker
        fig.canvas.draw_idle()  # Redraw the canvas

    # Link the update function to the slider
    slider.on_changed(update)

    # Show the interactive plot
    plt.tight_layout()
    plt.show()