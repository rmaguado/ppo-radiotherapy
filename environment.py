from scipy.spatial.transform import Rotation as R
import numpy as np

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple
import os

from generate_voxel_data import beam_voxels


class RadiotherapyGym(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, volume_shape=(64, 64, 64, 4), action_space_size=8):
        super().__init__()

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=volume_shape, dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(action_space_size,), dtype=np.float32
        )

        self.volume = np.random.rand(*volume_shape).astype(np.float32)
        self.current_state = self.volume.copy()
        self.done = False

    def step(self, action):
        raise NotImplementedError
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        raise NotImplementedError
        return observation, info

    def render(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError


def apply_rotation(
    direction: np.ndarray,
    rotation_vector: np.ndarray,
    max_slope: float,
    epsilon: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies a rotation to a direction vector using Rodrigues' rotation formula.
    Ensures that the slope of the z-component is within a given threshold.

    Args:
        direction (np.ndarray): The direction vector.
        rotation_vector (np.ndarray): The rotation vector.
        max_slope (float): The maximum slope of the z-component.
        epsilon (float): A small value to avoid division by zero.

    Returns:
        new_direction (np.ndarray): The new direction vector.
        overshoot (np.ndarray): The overshoot of the rotation.
    """
    norm_direction = np.linalg.norm(direction)
    if norm_direction > epsilon:
        direction = direction / norm_direction

    norm_rotation = np.linalg.norm(rotation_vector)
    if norm_rotation > epsilon:
        rotation_vector = rotation_vector / norm_rotation

    theta = np.linalg.norm(rotation_vector)
    if theta > epsilon:
        k = rotation_vector / theta
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        rotation_matrix = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    else:
        rotation_matrix = np.eye(3)

    new_direction = rotation_matrix @ direction

    xy_magnitude = np.linalg.norm(new_direction[:2]) + epsilon
    slope_z = new_direction[2] / xy_magnitude

    overshoot = np.array([0.0, 0.0, 0.0])
    if np.abs(slope_z) > max_slope:
        overshoot[2] = np.sign(slope_z) * (np.abs(slope_z) - max_slope)
        new_direction[2] -= overshoot[2] * xy_magnitude

    new_direction = new_direction / (np.linalg.norm(new_direction) + epsilon)

    return new_direction, overshoot


def apply_translation(
    position: np.ndarray, translation_vector: np.ndarray, bounds: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    translated_position = position + translation_vector
    bounded_new_position = np.clip(translated_position, bounds[0], bounds[1])
    overshoot = np.abs(translated_position - bounded_new_position)

    return bounded_new_position, overshoot


class RadiotherapyEnv:
    ACTION_SIZE = 8
    MAX_TIME_STEPS = 100
    MAX_SLOPE = 0.6
    LUNG_DOSE_THRESHOLD = 0.1
    TUMOUR_DOSE_THRESHOLD = 0.9
    LUNG_DOSE_REWARD = -1.0
    TUMOUR_DOSE_REWARD = 1.0

    LUNGS_ARRAY = np.load("./data/lungs.npy").astype(np.float32)
    TUMOUR_DIRS = [x for x in os.listdir("./data/tumours") if x.endswith(".npy")]

    def __init__(self):
        self.lungs = np.copy(RadiotherapyEnv.LUNGS_ARRAY)
        self.shape = self.lungs.shape
        self.tumours = None
        self.dose = None
        self.beams = []

        self.timestep = 0
        self.beam_position = None
        self.beam_direction = None
        self.done = False

        self.reset()

    def reset(self):
        self.load_data()
        self.reset_beam()
        self.reset_dose()
        self.timestep = 0
        self.done = False

    def load_data(self):
        self.tumours = np.zeros_like(self.lungs, dtype=np.float32)

        n_tumours = np.random.randint(1, 3)
        for _ in range(n_tumours):
            tumour_path = (
                f"./data/tumours/{np.random.choice(RadiotherapyEnv.TUMOUR_DIRS)}"
            )
            self.tumours += np.load(tumour_path).astype(np.float32)

    def reset_beam(self):
        self.beams = []
        self.beam_position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.beam_direction = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    def reset_dose(self):
        self.dose = np.zeros_like(self.lungs, dtype=np.float32)

    def add_beam(self, position, direction):
        self.dose += beam_voxels(self.lungs, position, direction)
        self.beams.append((position, direction))

    def map_translation(self, normalized_translation):
        """
        Maps the normalized translation to a translation vector.

        Args:
            normalized_translation (np.ndarray): The normalized translation vector. Range is [-1, 1].

        Returns:
            np.ndarray: The translation vector. Range is [0, shape - 1].
        """

        translation_bounds = np.array(self.shape) - 1

        return normalized_translation * translation_bounds

    def map_rotation(self, normalized_rotation):
        """
        Maps the normalized rotation to a rotation vector.

        Args:
            normalized_rotation (np.ndarray): The normalized rotation vector. Range is [-1, 1].

        Returns:
            np.ndarray: The rotation vector. Range is [-pi, pi].
        """

        rotation_bounds = np.pi
        rotation_vector = normalized_rotation * rotation_bounds

        return rotation_vector

    def tumour_dose_reward(self):
        raise NotImplementedError

    def lungs_dose_reward(self):
        raise NotImplementedError

    def overshoot_reward(self, translation_overshoot, rotation_overshoot):
        raise NotImplementedError

    def map_bool(self, beam_on):
        return beam_on > 0.0

    def step(self, action):
        normalized_translation = action[:3]
        normalized_rotation = action[3:6]
        is_beam_on = self.map_bool(action[6])
        finished = self.map_bool(action[7])

        if finished:
            self.done = True
            return

        translation = self.map_translation(normalized_translation)
        rotation = self.map_rotation(normalized_rotation)

        new_position, overshoot_translation = apply_translation(
            self.beam_position, translation, np.array(self.shape) - 1
        )
        new_direction, overshoot_rotation = apply_rotation(
            self.beam_direction, rotation, RadiotherapyEnv.MAX_SLOPE
        )

        self.beam_position = new_position
        self.beam_direction = new_direction

        if is_beam_on:
            self.add_beam(self.beam_position, self.beam_direction)

        self.timestep += 1


if __name__ == "__main__":
    pass
