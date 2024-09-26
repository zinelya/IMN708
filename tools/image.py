import nibabel as nib
import nibabel.orientations as orient
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
import tools.io

def reorient_data_rsa(image):
    """
    Check the NIfTI orientation, and flip to  'RPS' if needed.
    :param image: NIfTI file
    :param data: array file
    :return: array after flipping
    """
    data = image.get_fdata()
        
    x, y, z = nib.aff2axcodes(image.affine)
    # print(x, y, z, image.affine)
    
    if x != 'R':
        data = np.flip(data, axis=0)
    if y != 'A':
        data = np.flip(data, axis=1)
    if z != 'S':
        data = np.flip(data, axis=2)

    return data

def afficher_image(in_image, axe, titre='Image'):
    # Check the validity of input image
    image = tools.io.check_valid_image(in_image)

    # Stop the function if the image is invalid.
    if image:
        data = reorient_data_rsa(image)
        voxel_sizes = image.header.get_zooms()
        
        # Check if the image is 4D
        is_4d = data.ndim == 4
        current_time = 0
        num_timepoints = data.shape[3] if is_4d else 1
        
        # Initialize the figure and axis
        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.25)  # Adjust space for the time slider
        
        # Set the initial slice to the middle of the chosen axis
        current_slice = data.shape[axe] // 2
        total_slices = data.shape[axe]
        
        # Function to update the displayed image based on the current slice and time point
        def update_display():
            ax.clear()  # Clear the previous image
            if axe == 0:  # Sagittal
                img_slice = data[current_slice, :, :, current_time] if is_4d else data[current_slice, :, :]
            elif axe == 1:  # Coronal
                img_slice = data[:, current_slice, :, current_time] if is_4d else data[:, current_slice, :]
            elif axe == 2:  # Axial
                img_slice = data[:, :, current_slice, current_time] if is_4d else data[:, :, current_slice]
            
            # Transpose data for right 
            transposed_slice = img_slice.T
            
            if axe > 0:
                transposed_slice = np.fliplr(transposed_slice)

            # Determine aspect ratio for the current axis
            aspect_ratios = [(voxel_sizes[1] / voxel_sizes[2]),  # Sagittal (X-axis)
                            (voxel_sizes[0] / voxel_sizes[2]),  # Coronal (Y-axis)
                            (voxel_sizes[0] / voxel_sizes[1])]  # Axial (Z-axis)

            # Map the axes and set the title based on the projection type
            titles = ['Sagittal', 'Coronal', 'Axial']
            ax.imshow(transposed_slice, cmap='gray', origin='lower', aspect=aspect_ratios[axe])
            ax.set_title(f'{titles[axe]} slice {current_slice}')
            
            fig.canvas.draw_idle()  # Redraw the canvas with the updated image
        
        # Function to handle scroll events for slice navigation
        def on_scroll(event):
            nonlocal current_slice
            if event.button == 'up':
                if current_slice < total_slices - 1:
                    current_slice += 1
            elif event.button == 'down':
                if current_slice > 0:
                    current_slice -= 1
                    
            update_display()  # Update the display with the new slice
        
        # Function to handle time slider changes
        def on_time_slider(val):
            nonlocal current_time
            current_time = int(val)
            update_display()
        
        # Connect the scroll event to the figure for slice navigation
        fig.canvas.mpl_connect('scroll_event', on_scroll)

        # Create a time slider if the image is 4D
        if is_4d and num_timepoints > 1:
            ax_time = plt.axes([0.25, 0.1, 0.65, 0.03])  # Position of the time slider
            time_slider = Slider(ax_time, 'Time', 0, num_timepoints - 1, valinit=0, valstep=1)
            time_slider.on_changed(on_time_slider)

        # Display the initial slice
        update_display()

        plt.show()
        
def plot_histogram(in_image, bin, in_mask):
    # Check the validity of input image
    image = tools.io.check_valid_image(in_image)

    # Stop the function if the image is invalid.
    if image:
        data = image.get_fdata()
        flattened_array = data.flatten()

        # Step 2: Plot the histogram with 10 bins
        plt.hist(flattened_array, bins=bin, density=True, edgecolor='black')
        plt.title('Intensity histogram')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.show()
        
        if in_mask:
            mask = tools.io.check_valid_image(in_mask)
            
            if image.shape == mask.shape:
                mask_data = mask.get_fdata()
                unique_labels = np.unique(mask_data[mask_data != 0])
                
                for label in unique_labels:
                    roi = data[mask_data == label]
                    flattened_array = roi.flatten()

                    # Step 2: Plot the histogram with 10 bins
                    plt.hist(flattened_array, bins=bin, density=True, edgecolor='black')
                    plt.title(f'Intensity histogram for segment {int(label)}')
                    plt.xlabel('Value')
                    plt.ylabel('Frequency')
                    plt.show()
            else:   
                print("The shapes of orginal image and mask are different.")
