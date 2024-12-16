import numpy as np
from tools.tensor import construct_D_matrix


def upsample_seeding_points(seeding_points, num_points=10):
    """
    Generate random points within each voxel defined by seeding indices.
    """
    random_offsets = np.random.uniform(0, 1, size=(len(seeding_points), num_points, 3))
    random_points = seeding_points[:, np.newaxis, :] + random_offsets
    return random_points.reshape(-1, 3)

def trilinear_interpolate(data, point, order = 'mrtrix'):
    """
    Perform trilinear interpolation for a tensor or vector at a given 3D point in a 4D array.

    Args:
        data (numpy.ndarray): A 4D array of shape (Nx, Ny, Nz, T), where T can represent the size of a tensor or a 3D vector.
        point (tuple): A tuple (x, y, z) representing the continuous point in the 3D space.
        order (str): Format of the tensor data ("fsl" or "mrtrix").

    Returns:
        numpy.ndarray: The interpolated tensor, vector, or eigenvector at the given point.
    """
    x, y, z = point
    x0, x1 = int(np.floor(x)), int(np.ceil(x))
    y0, y1 = int(np.floor(y)), int(np.ceil(y))
    z0, z1 = int(np.floor(z)), int(np.ceil(z))


    x0, x1 = np.clip([x0, x1], 0, data.shape[0] - 1)
    y0, y1 = np.clip([y0, y1], 0, data.shape[1] - 1)
    z0, z1 = np.clip([z0, z1], 0, data.shape[2] - 1)

    xd, yd, zd = x - x0, y - y0, z - z0


    c000, c100 = data[x0, y0, z0], data[x1, y0, z0]
    c010, c110 = data[x0, y1, z0], data[x1, y1, z0]
    c001, c101 = data[x0, y0, z1], data[x1, y0, z1]
    c011, c111 = data[x0, y1, z1], data[x1, y1, z1]


    c00 = c000 * (1 - xd) + c100 * xd
    c10 = c010 * (1 - xd) + c110 * xd
    c01 = c001 * (1 - xd) + c101 * xd
    c11 = c011 * (1 - xd) + c111 * xd

    c0 = c00 * (1 - yd) + c10 * yd
    c1 = c01 * (1 - yd) + c11 * yd

    interpolated_result = c0 * (1 - zd) + c1 * zd


    # peaks
    if data.shape[-1] == 3:
        norm = np.linalg.norm(interpolated_result)
        return interpolated_result / norm if norm > 0 else np.zeros(3)
    
    # DTI tensors
    elif data.shape[-1] == 6:  # Assuming 6 components for symmetric tensors
        D_voxel = construct_D_matrix(interpolated_result, 0, 0, 0, order)
        vals, vecs = np.linalg.eigh(D_voxel)
        principal_vector = vecs[:, np.argmax(vals)]
        return principal_vector / np.linalg.norm(principal_vector)
    
    else:
        raise ValueError(f"Unsupported data shape for interpolation: {data.shape[-1]}")


def compute_angle(v1, v2):
    """
    Compute the angle between two vectors in degrees.
    """
    v1_norm, v2_norm = np.linalg.norm(v1), np.linalg.norm(v2)
    if v1_norm == 0 or v2_norm == 0:
        return 180.0  # Invalid vectors
    return np.degrees(np.arccos(np.clip(np.dot(v1 / v1_norm, v2 / v2_norm), -1.0, 1.0)))

def compute_streamline_length(streamline):
    """
    Compute the length of a streamline.
    """
    diffs = np.diff(streamline, axis=0)
    return np.sum(np.linalg.norm(diffs, axis=1))

# --- Tracking Functions ---
def track_single_direction(peaks, stop_mask, seed, direction, step_size, angle_threshold, max_length):
    """
    Perform tracking in a single direction (either forward or backward).
    """
    tract = [seed]
    current_position = np.array(seed, dtype=float)
    previous_vector = None

    while compute_streamline_length(tract) <= max_length:
        x, y, z = map(int, np.round(current_position))
        if not (0 <= x < stop_mask.shape[0] and 0 <= y < stop_mask.shape[1] and 0 <= z < stop_mask.shape[2]):
            break  # Out of bounds
        if not stop_mask[x, y, z]:
            break  # Outside white matter mask

        principal_vector = trilinear_interpolate(peaks, current_position)
        if np.linalg.norm(principal_vector) == 0:
            break  # Undefined principal vector

        if previous_vector is not None:
            angle = compute_angle(previous_vector, principal_vector * direction)
            if angle > angle_threshold:
                break  # Angle threshold exceeded

        next_position = current_position + step_size * principal_vector * direction
        tract.append(next_position.tolist())
        previous_vector = principal_vector * direction
        current_position = next_position

    return tract

def track_streamlines(peaks, stop_mask, seed_mask, seed_density=5, step_size=1, angle_threshold=40, min_length=10.0, max_length=100.0):
    """
    Perform deterministic streamline tracking.
    """
    seeding_indices = np.argwhere(seed_mask)
    seed_points = upsample_seeding_points(seeding_indices, num_points=seed_density)
    tracts = []
    for seed in seed_points:
        forward_tract = track_single_direction(peaks, stop_mask, seed, direction=1, step_size=step_size, angle_threshold=angle_threshold, max_length=max_length)
        backward_tract = track_single_direction(peaks, stop_mask, seed, direction=-1, step_size=step_size, angle_threshold=angle_threshold, max_length=max_length)
        combined_tract = backward_tract[::-1] + forward_tract[1:]
        if compute_streamline_length(combined_tract) >= min_length:
            tracts.append(np.array(combined_tract))

    return tracts