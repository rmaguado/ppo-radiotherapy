import numpy as np

from stable_baselines3.common.env_checker import check_env
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import datetime

from visualize_voxel import view_observation_slices
from graphics import beam_voxels, create_animation, create_scene
from transforms import apply_rotation, apply_translation


class RadiotherapyEnv(gym.Env):
    ACTION_SIZE = 8
    MAX_TIME_STEPS = 100
    MAX_SLOPE = 0.6
    LUNG_DOSE_THRESHOLD = 0.1
    TUMOUR_DOSE_THRESHOLD = 0.9
    LUNG_DOSE_REWARD = -1.0
    TUMOUR_DOSE_REWARD = -100.0
    OVERSHOOT_TRANSLATION_REWARD = -0.1
    OVERSHOOT_ROTATION_REWARD = -0.1

    TUMOUR_DIRS = [x for x in os.listdir("./data/tumours") if x.endswith(".npy")]
    LUNGS_ARRAY = np.load("./data/lungs.npy").astype(np.float32)
    LUNG_SHAPE = np.array(LUNGS_ARRAY.shape)
    TRANSLATION_BOUNDS = LUNG_SHAPE - 1
    OBSERVATION_SHAPE = (4, LUNG_SHAPE[0], LUNG_SHAPE[1], LUNG_SHAPE[2])

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self):
        super().__init__()

        self.lungs = np.copy(self.LUNGS_ARRAY)
        self.tumours = None
        self.tumours_meta = []
        self.dose = None
        self.beams = []

        self.export_gif = False

        self.t = 0
        self.beam_position = None
        self.beam_direction = None
        self.done = False

        self.reset()

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=self.OBSERVATION_SHAPE, dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.ACTION_SIZE,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        self.reset_tumours()
        self.reset_beam()
        self.reset_dose()
        self.t = 0
        self.done = False

        return self.observation(), {}

    def reset_tumours(self):
        self.tumours = np.zeros_like(self.lungs, dtype=np.float32)

        n_tumours = np.random.randint(1, 3)
        for _ in range(n_tumours):
            tumour_filename = np.random.choice(self.TUMOUR_DIRS)
            tumour_attrs = tumour_filename.split(".npy")[0].split("_")
            tumour_position = np.array(tumour_attrs[:3], dtype=np.float32)
            tumour_radius = float(tumour_attrs[3])
            tumour_path = f"./data/tumours/{tumour_filename}"
            self.tumours += np.load(tumour_path).astype(np.float32)
            self.tumours_meta.append((tumour_position, tumour_radius))

    def reset_beam(self):
        self.beams = []
        self.beam_position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.beam_direction = np.array([1.0, 1.0, 1.0], dtype=np.float32)

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

        return normalized_translation * self.TRANSLATION_BOUNDS

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
        tumour_dose = self.dose * self.tumours
        threshold_mask = tumour_dose > self.TUMOUR_DOSE_THRESHOLD
        above_threshold_dose = np.sum(threshold_mask * tumour_dose)
        dose_ratio = above_threshold_dose / np.sum(self.tumours)

        return (1 - dose_ratio) * self.TUMOUR_DOSE_REWARD

    def lungs_dose_reward(self):
        lungs_dose = self.dose * self.lungs * (1 - self.tumours)
        threshold_mask = lungs_dose > self.LUNG_DOSE_THRESHOLD
        above_threshold_dose = np.sum(threshold_mask * lungs_dose)

        return above_threshold_dose * self.LUNG_DOSE_REWARD

    def dose_reward(self):
        return self.tumour_dose_reward() + self.lungs_dose_reward()

    def overshoot_reward(self, translation_overshoot, rotation_overshoot):
        translation_reward = (
            -np.sum(translation_overshoot) * self.OVERSHOOT_TRANSLATION_REWARD
        )
        rotation_reward = -np.sum(rotation_overshoot) * self.OVERSHOOT_ROTATION_REWARD

        return translation_reward + rotation_reward

    def map_bool(self, beam_on):
        return beam_on > 0.0

    def step(self, action):
        normalized_translation = action[:3]
        normalized_rotation = action[3:6]
        is_beam_on = self.map_bool(action[6])
        done = self.map_bool(action[7])

        if done:
            self.done = True
            self.t += 1
            reward = self.dose_reward()
            return self.observation(), reward, True, False, {}

        translation = self.map_translation(normalized_translation)
        rotation = self.map_rotation(normalized_rotation)

        new_position, overshoot_translation = apply_translation(
            self.beam_position, translation, self.TRANSLATION_BOUNDS
        )
        new_direction, overshoot_rotation = apply_rotation(
            self.beam_direction, rotation, self.MAX_SLOPE
        )

        self.beam_position = new_position
        self.beam_direction = new_direction

        if is_beam_on:
            self.add_beam(self.beam_position, self.beam_direction)

        reward = self.dose_reward() + self.overshoot_reward(
            overshoot_translation, overshoot_rotation
        )

        self.t += 1

        return self.observation(), reward, False, False, {}

    def observation(self):
        current_beam = beam_voxels(self.lungs, self.beam_position, self.beam_direction)
        stacked = np.stack([self.lungs, self.tumours, self.dose, current_beam], axis=0)

        return np.clip(stacked, 0.0, 1.0)

    def export_animation(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        create_animation(
            self.tumours_meta,
            self.beams,
            self.LUNG_SHAPE,
            filename=f"animations/{timestamp}.gif",
            export_gif=self.export_gif,
            window=True,
        )

    def render(self):
        create_scene(
            self.tumours_meta,
            self.beams,
            self.LUNG_SHAPE,
        )

    def inspect_observation(self):
        observation = self.observation()
        view_observation_slices(observation)

    def close(self):
        pass


def test_check_env():
    print("Initialising environment...")
    env = RadiotherapyEnv()
    print("Checking environment...")
    result = check_env(env)
    print("Environment check result:", result)
    env.close()


def test_observation_render():
    env = RadiotherapyEnv()

    env.step(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]))
    env.step(np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]))
    env.step(np.array([0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.0]))
    env.step(np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]))
    env.step(np.array([0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 1.0, 0.0]))

    env.inspect_observation()
    env.render()
    env.close()


def test_observation_shape():
    env = RadiotherapyEnv()
    obs = env.observation()
    print("Observation shape:", obs.shape)
    env.close()


if __name__ == "__main__":
    test_observation_render()
