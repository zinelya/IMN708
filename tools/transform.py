import numpy as np

def trans_rigide(grid, theta, omega, phi, p, q, r):
    """
    Returns the homogeneous transformation matrix that performs:
    i.   a rotation by angle theta around the x-axis,
    ii.  a rotation by angle omega around the y-axis,
    iii. a rotation by angle phi around the z-axis,
    iv.  a translation by the vector t = (p, q, r).
    
    Args:
    grid : numpy array (N, 3)
        Set of points to transform.
    theta : float
        Rotation angle around the x-axis (in radians).
    omega : float
        Rotation angle around the y-axis (in radians).
    phi : float
        Rotation angle around the z-axis (in radians).
    p, q, r : float
        Translation vector components.
    
    Returns:
    trans_grid : numpy array (N, 3)
        Transformed grid of points after applying the rotation and translation.
    """

    # Convert angles from degrees to radians
    theta_rad = np.radians(theta)
    omega_rad = np.radians(omega)
    phi_rad = np.radians(phi)

    # Rotation matrix around the x-axis
    Rx = np.array([[1, 0, 0, 0],
                   [0, np.cos(theta_rad), -np.sin(theta_rad), 0],
                   [0, np.sin(theta_rad), np.cos(theta_rad), 0],
                   [0, 0, 0, 1]])

    # Rotation matrix around the y-axis
    Ry = np.array([[np.cos(omega_rad), 0, np.sin(omega_rad), 0],
                   [0, 1, 0, 0],
                   [-np.sin(omega_rad), 0, np.cos(omega_rad), 0],
                   [0, 0, 0, 1]])

    # Rotation matrix around the z-axis
    Rz = np.array([[np.cos(phi_rad), -np.sin(phi_rad), 0, 0],
                   [np.sin(phi_rad), np.cos(phi_rad), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])

    # Translation matrix
    T = np.array([[1, 0, 0, p],
                  [0, 1, 0, q],
                  [0, 0, 1, r],
                  [0, 0, 0, 1]])

    # Combine all transformations into a single rigid transformation matrix
    rigid_transform = T @ Rz @ Ry @ Rx

    # Convert grid points to homogeneous coordinates (add a 1 for each point)
    N = grid.shape[0]
    grid_homogeneous = np.hstack((grid, np.ones((N, 1))))

    # Apply the rigid transformation
    trans_grid_homogeneous = (rigid_transform @ grid_homogeneous.T).T

    # Return the transformed points in 3D (remove the homogeneous coordinate)
    trans_grid = trans_grid_homogeneous[:, :3]
    
    return trans_grid

def similitude(grid, s):
    """
    Applies a scaling (homothety) transformation to a 3D grid of points, centered around the first point in the grid.
    
    Args:
    grid : numpy array (N, 3)
        Set of 3D points to be scaled.
    s : float
        Scaling factor (homothety ratio).
    
    Returns:
    scaled_grid : numpy array (N, 3)
        The grid of points after applying the scaling transformation around the first point.
    """
    # Take the first point in the grid as the reference for scaling
    first_point = grid[0, :]
    
    # Translate the grid to move the first point to the origin
    translated_grid = grid - first_point
    
    # Apply scaling around the origin
    scaled_translated_grid = translated_grid * s
    
    # Translate the grid back to the original first point
    scaled_grid = scaled_translated_grid + first_point
    
    return scaled_grid

def trans_with_affine_matrix(grid, affine_matrix):
    # Convert grid points to homogeneous coordinates (add a 1 for each point)
    N = grid.shape[0]
    grid_homogeneous = np.hstack((grid, np.ones((N, 1))))

    # Apply the rigid transformation
    trans_grid_homogeneous = (affine_matrix @ grid_homogeneous.T).T

    # Return the transformed points in 3D (remove the homogeneous coordinate)
    print(affine_matrix)
    R = affine_matrix[:3, :3]
    translation = affine_matrix[:3, 3]
    U, S, Vt = np.linalg.svd(R)
    print("U (Rotation matrix from left side):")
    print(U)
    print("\nSigma (Diagonal matrix - Scaling factors):")
    print(np.diag(S))
    print("\nVT (Rotation matrix from right side):")
    print(Vt)
    print("\nTranslation vector:")
    print(translation)

    return trans_grid_homogeneous
