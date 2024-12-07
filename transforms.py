import numpy as np
from scipy.spatial.transform import Rotation as R

from typing import Tuple


def apply_rotation(
    initial_direction: np.ndarray, rotation_vector: np.ndarray, min_angle: float
) -> Tuple[np.ndarray, float]:
    """
    Applies a set of rotations to a 3D direction vector.
    The slope along the z-axis should be at least min_angle.

    Args:
        initial_direction (np.ndarray): The input direction vector (z, x, y).
        rotation_vector (np.ndarray): The rotation vector in radians (around z, x, y axes).
        min_angle (float): The minimum angle in radians relative to the z-axis.

    Returns:
        np.ndarray: The new direction vector after rotation.
        float: The overshoot angle in radians if the angle exceeded min_angle, else 0.
    """
    direction = initial_direction / np.linalg.norm(initial_direction)

    rotation = R.from_rotvec(rotation_vector)
    rotated_direction = rotation.apply(direction)
    rotated_direction /= np.linalg.norm(rotated_direction)

    z_component = np.clip(rotated_direction[0], -1.0, 1.0)
    z_component_sign = np.sign(z_component)
    angle_with_z_axis = np.arccos(z_component)
    if z_component_sign < 0:
        angle_with_z_axis = np.pi - angle_with_z_axis

    if abs(angle_with_z_axis) < min_angle:
        target_z = z_component_sign * np.cos(min_angle)
        target_xy_magnitude = np.sqrt(1 - target_z**2)

        xy_projection = rotated_direction[1:]
        if np.linalg.norm(xy_projection) < 1e-8:
            xy_projection = np.array([1.0, 0.0])
        else:
            xy_projection /= np.linalg.norm(xy_projection)

        new_direction = np.array(
            [
                target_z,
                xy_projection[0] * target_xy_magnitude,
                xy_projection[1] * target_xy_magnitude,
            ]
        )
    else:
        new_direction = rotated_direction

    new_direction /= np.linalg.norm(new_direction)

    overshoot = max(0, min_angle - angle_with_z_axis)

    return new_direction, overshoot


def apply_translation(
    position: np.ndarray, translation_vector: np.ndarray, bounds: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    translated_position = position + translation_vector
    bounded_new_position = np.clip(translated_position, 0, bounds)
    overshoot = np.abs(translated_position - bounded_new_position)

    return bounded_new_position, overshoot


def test_rotation():
    direction = np.array([0.0, 1.0, 0.0])
    rotation_vector = np.array([0.0, 0.0, -np.pi / 3])
    max_angle = np.pi / 4

    apply_rotation(direction, rotation_vector, max_angle)


if __name__ == "__main__":
    test_rotation()
