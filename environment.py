from scipy.spatial.transform import Rotation as R
import numpy as np


def apply_rotation(
    direction: np.ndarray, rotation_vector: np.ndarray, max_slope: float, epsilon=1e-6
) -> np.ndarray:
    """
    Rotate the direction vector while enforcing slope constraints along the axial dimension.

    Args:
        direction (np.ndarray): The original direction vector (must be a unit vector).
        rotation_vector (np.ndarray): A 3D array [Delta_theta_x, Delta_theta_y, Delta_theta_z].
        max_slope (float): The maximum allowed absolute slope (Z component / vector norm).
        epsilon (float): In case the direction is the z-vector.

    Returns:
        np.ndarray: The new constrained direction vector.
    """
    direction = direction / np.linalg.norm(direction)

    quaternion = R.from_rotvec(rotation_vector)
    new_direction = quaternion.apply(direction)

    z_component = new_direction[0]
    z_component_slope = np.abs(z_component) / np.linalg.norm(new_direction)

    if np.abs(z_component_slope) > max_slope:
        if np.abs(new_direction[1]) == 0 and np.abs(new_direction[2]) == 0:
            new_direction = np.array(
                [max_slope * np.sign(z_component), (1 - max_slope), 0]
            )
        else:
            new_direction = np.array(
                [
                    (1 - max_slope) * np.sign(z_component),
                    new_direction[1],
                    new_direction[2],
                ]
            )
        new_direction = new_direction / np.linalg.norm(new_direction)

    return new_direction


def apply_translation(
    position: np.ndarray, translation_vector: np.ndarray, bounds: np.ndarray
) -> np.ndarray:
    """
    Translate the position vector while enforcing position constraints.

    Args:
        position (np.ndarray): The original position vector.
        translation_vector (np.ndarray): A 3D array [Delta_x, Delta_y, Delta_z].
        bounds (np.ndarray): A 2x3 array [[x_min, y_min, z_min], [x_max, y_max, z_max]]
                             defining the allowed position range.

    Returns:
        np.ndarray: The new constrained position vector.
    """
    new_position = position + translation_vector

    new_position = np.clip(new_position, bounds[0], bounds[1])

    return new_position


class RadiotherapyEnvironment:
    def __init__(self):
        # voxel images
        self.lungs = None
        self.tumours = None
        self.dose = None
        self.beams = []


if __name__ == "__main__":
    pass
