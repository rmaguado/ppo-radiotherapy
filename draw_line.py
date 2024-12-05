import numpy as np


def beam_voxels(base_matrix, position, direction, epsilon=1e-6):
    """
    Discretizes a line in a 3D matrix using Xiaolin Wu's Algorithm with numerical stability.

    Parameters:
        base_matrix (np.ndarray): The 3D matrix defining the space.
        position (tuple): The translation vector of the beam as a 3D tuple (x, y, z).
        direction (tuple): The direction vector of the beam.
        epsilon (float): Tolerance for numerical errors in the direction vector.

    Returns:
        np.ndarray: The 3D matrix with the discretized beam applied.
    """
    output = np.zeros_like(base_matrix, dtype=np.float32)

    position = np.array(position, dtype=np.float32)
    direction = np.array(direction, dtype=np.float32)

    norm = np.linalg.norm(direction)
    if norm < epsilon:
        raise ValueError("Direction vector magnitude is too small.")
    direction /= norm

    grid_size = base_matrix.shape

    t_min = []
    t_max = []
    for i in range(3):
        if abs(direction[i]) > epsilon:
            t1 = (-position[i]) / direction[i]
            t2 = (grid_size[i] - 1 - position[i]) / direction[i]
            t_entry_i = min(t1, t2)
            t_exit_i = max(t1, t2)
        else:
            if position[i] < 0 or position[i] > grid_size[i] - 1:
                return output
            t_entry_i = -np.inf
            t_exit_i = np.inf
        t_min.append(t_entry_i)
        t_max.append(t_exit_i)
    t_entry = max(t_min)
    t_exit = min(t_max)
    if t_entry > t_exit:
        return output

    dir_abs = np.abs(direction)
    dominant_axis = np.argmax(dir_abs)
    other_axes = [i for i in range(3) if i != dominant_axis]

    step = 1 if direction[dominant_axis] > 0 else -1

    start_voxel = int(
        np.floor(position[dominant_axis] + t_entry * direction[dominant_axis])
    )
    end_voxel = int(
        np.floor(position[dominant_axis] + t_exit * direction[dominant_axis])
    )

    intery = position[other_axes[0]] + t_entry * direction[other_axes[0]]
    interz = position[other_axes[1]] + t_entry * direction[other_axes[1]]

    gradient_y = direction[other_axes[0]] / (direction[dominant_axis] + epsilon)
    gradient_z = direction[other_axes[1]] / (direction[dominant_axis] + epsilon)

    x = start_voxel
    while (x - end_voxel) * step <= 0:
        y = intery
        z = interz

        idx = [0, 0, 0]
        idx[dominant_axis] = x

        y_floor = int(np.floor(y))
        y_frac = y - y_floor
        idx[other_axes[0]] = y_floor

        z_floor = int(np.floor(z))
        z_frac = z - z_floor
        idx[other_axes[1]] = z_floor

        for dy in [0, 1]:
            for dz in [0, 1]:
                weight = (1 - y_frac) if dy == 0 else y_frac
                weight *= (1 - z_frac) if dz == 0 else z_frac
                ix = idx[0]
                iy = idx[1] + dy
                iz = idx[2] + dz
                if (
                    0 <= ix < grid_size[0]
                    and 0 <= iy < grid_size[1]
                    and 0 <= iz < grid_size[2]
                ):
                    output[ix, iy, iz] += weight

        intery += gradient_y * step
        interz += gradient_z * step
        x += step

    return output
