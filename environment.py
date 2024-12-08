import numpy as np

from stable_baselines3.common.env_checker import check_env
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import datetime

from visualize_voxel import view_observation_slices
from draw_line import beam_voxels
from transforms import apply_rotation, apply_translation


class RadiotherapyEnv(gym.Env):
    ACTION_SIZE = 6
    MAX_TIME_STEPS = 100
    MIN_ANGLE_Z = np.pi / 4
    BEAM_DOSE = 0.1
    LUNG_DOSE_THRESHOLD = 0.2
    LUNG_DOSE_REWARD = -1.0
    TUMOUR_DOSE_REWARD = 1.0
    OVERSHOOT_TRANSLATION_REWARD = -1.0
    OVERSHOOT_ROTATION_REWARD = -1.0
    DISTANCE_TO_TUMOUR_REWARD = -1.0
    STILL_PENALTY_REWARD = -1.0
    MOVEMENT_THRESHOLD = 0.05
    ROTATION_THRESHOLD = 0.05

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

        self.max_episode_steps = self.MAX_TIME_STEPS

    def export_trajectory(self, filename):
        np.savez_compressed(
            filename,
            tumours=self.tumours,
            dose=self.dose,
            beams=self.beams,
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

        n_tumours = 1
        for _ in range(n_tumours):
            tumour_filename = np.random.choice(self.TUMOUR_DIRS)
            tumour_attrs = tumour_filename.split(".npy")[0].split("_")
            tumour_position = np.array(tumour_attrs[:3], dtype=np.float32)
            tumour_radius = float(tumour_attrs[3])
            tumour_path = f"./data/tumours/{tumour_filename}"
            self.tumours += np.load(tumour_path).astype(np.float32)
            self.tumours_meta.append((tumour_position, tumour_radius))
        self.tumours = np.clip(self.tumours, 0.0, 1.0)

    def reset_beam(self):
        self.beams = []
        self.beam_position = np.array(self.LUNG_SHAPE) / 2
        self.beam_direction = np.array([0.0, 1.0, 0.0])

    def reset_dose(self):
        self.dose = np.zeros_like(self.lungs, dtype=np.float32)

    def add_beam(self, position, direction):
        self.dose += beam_voxels(self.lungs, position, direction) * self.BEAM_DOSE
        self.dose = np.clip(self.dose, 0.0, 1.0)
        self.beams.append((position, direction))

    def map_translation(self, normalized_translation):
        """
        Maps the normalized translation to a translation vector.

        Args:
            normalized_translation (np.ndarray): The normalized translation vector. Range is [-1, 1].

        Returns:
            np.ndarray: The translation vector. Range is [0, shape - 1].
        """
        clipped_translation = np.clip(normalized_translation, -1.0, 1.0)
        return clipped_translation * self.TRANSLATION_BOUNDS

    def map_rotation(self, normalized_rotation):
        """
        Maps the normalized rotation to a rotation vector.

        Args:
            normalized_rotation (np.ndarray): The normalized rotation vector. Range is [-1, 1].

        Returns:
            np.ndarray: The rotation vector. Range is [-pi, pi].
        """

        rotation_bounds = np.pi
        clipped_rotation = np.clip(normalized_rotation, -1.0, 1.0)
        rotation_vector = clipped_rotation * rotation_bounds

        return rotation_vector

    def distance_to_tumour(self):
        tumour = np.stack(np.where(self.tumours == 1.0), axis=-1)
        beam_position = np.array(self.beam_position)
        distances = np.linalg.norm(tumour - beam_position, axis=1)
        return np.min(distances)

    def distance_to_tumour_reward(self):
        distance = self.distance_to_tumour()
        relative_distance = distance / np.max(self.LUNG_SHAPE)
        return relative_distance * self.DISTANCE_TO_TUMOUR_REWARD

    def tumour_dose_reward(self):
        tumour_dose = self.dose * self.tumours
        total_tumour_dose = np.sum(tumour_dose)
        total_tumour = np.sum(self.tumours)

        tumour_dose_reward = total_tumour_dose / total_tumour * self.TUMOUR_DOSE_REWARD

        return tumour_dose_reward

    def lungs_dose_reward(self):
        lungs_mask = self.lungs * (1 - self.tumours)
        lungs_dose = self.dose * lungs_mask
        threshold_mask = lungs_dose > self.LUNG_DOSE_THRESHOLD
        above_threshold_count = np.sum(threshold_mask)
        total_lung = np.sum(lungs_mask)

        lung_dose_reward = above_threshold_count / total_lung * self.LUNG_DOSE_REWARD

        return lung_dose_reward

    def overshoot_reward(self, translation_overshoot, rotation_overshoot):
        translation_reward = (
            np.sum(translation_overshoot) * self.OVERSHOOT_TRANSLATION_REWARD
        )
        rotation_reward = np.sum(rotation_overshoot) * self.OVERSHOOT_ROTATION_REWARD

        return translation_reward + rotation_reward

    def still_penalty_reward(self, translation, rotation):
        translation_magnitude = np.linalg.norm(translation)
        rotation_magnitude = np.linalg.norm(rotation)

        if (
            translation_magnitude < self.MOVEMENT_THRESHOLD
            and rotation_magnitude < self.ROTATION_THRESHOLD
        ):
            return self.STILL_PENALTY_REWARD
        return 0.0

    def is_fully_irradiated(self):
        tumour_dose = self.dose * self.tumours
        total_tumour_dose = np.sum(tumour_dose)
        total_tumour = np.sum(self.tumours)

        if total_tumour_dose / total_tumour >= 0.99:
            return True
        return False

    def step(self, action):
        self.t += 1

        normalized_translation = action[:3]
        normalized_rotation = action[3:6]

        translation = self.map_translation(normalized_translation)
        rotation = self.map_rotation(normalized_rotation)

        new_position, overshoot_translation = apply_translation(
            self.beam_position, translation, self.TRANSLATION_BOUNDS
        )
        new_direction, overshoot_rotation = apply_rotation(
            self.beam_direction, rotation, self.MIN_ANGLE_Z
        )

        self.beam_position = new_position
        self.beam_direction = new_direction

        self.add_beam(self.beam_position, self.beam_direction)

        tumour_dose_reward = self.tumour_dose_reward()
        lungs_dose_reward = self.lungs_dose_reward()
        overshoot_reward = self.overshoot_reward(
            overshoot_translation, overshoot_rotation
        )
        still_penalty_reward = self.still_penalty_reward(translation, rotation)
        distance_to_tumour_reward = self.distance_to_tumour_reward()

        reward = (
            tumour_dose_reward
            + lungs_dose_reward
            + overshoot_reward
            + still_penalty_reward
            + distance_to_tumour_reward
        )

        self.done = self.is_fully_irradiated() or self.t >= self.MAX_TIME_STEPS

        info = {
            "reward_components": {
                "tumour": tumour_dose_reward,
                "lung": lungs_dose_reward,
                "overshoot": overshoot_reward,
                "movement": still_penalty_reward,
                "distance_to_tumour": distance_to_tumour_reward,
            },
            "beam_position": {
                "translation": list(new_position),
                "rotation": list(new_direction),
            },
            "doses": {
                "tumour": float(np.sum(self.dose * self.tumours)),
                "lung": float(np.sum(self.dose * self.lungs)),
            },
            "overshoot": {
                "translation": list(overshoot_translation),
                "rotation": overshoot_rotation,
            },
        }

        return self.observation(), reward, self.done, False, info

    def get_current_view(self):
        current_beam = beam_voxels(self.lungs, self.beam_position, self.beam_direction)
        horizontal_beam_center = beam_voxels(
            self.lungs, self.beam_position, np.array([1.0, 0.0, 0.0])
        )
        return current_beam + horizontal_beam_center

    def observation(self):
        current_beam_view = self.get_current_view()
        stacked = np.stack(
            [self.lungs, self.tumours, self.dose, current_beam_view], axis=0
        )

        return np.clip(stacked, 0.0, 1.0)

    def export_animation(self):
        from graphics import create_animation

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
        from graphics import create_scene

        create_scene(
            self.tumours_meta,
            self.beams,
            self.LUNG_SHAPE,
        )

    def inspect_observation(self):
        observation = self.observation()
        view_observation_slices(observation, axis=0)

    def close(self):
        pass


def test_check_env():
    print("Initialising environment...")
    env = RadiotherapyEnv()
    print("Checking environment...")
    result = check_env(env)
    print("Environment check result:", result)
    env.close()


def human_play():
    env = RadiotherapyEnv()

    print("Total tumour volume:", np.sum(env.tumours))
    print("Total lung volume:", np.sum(env.lungs))

    done = False
    while not done:
        # env.inspect_observation()
        env.render()

        action_raw = input("Enter action: ")
        if action_raw == "q":
            done = True
        else:
            action = np.array([float(x) for x in action_raw.split(",")])
            _, _, _, _, info = env.step(action)
            print("Info:")
            print(info)

    env.export_trajectory("trajectory.npz")
    env.close()


def test_observation_shape():
    env = RadiotherapyEnv()
    obs = env.observation()
    print("Observation shape:", obs.shape)
    env.close()


if __name__ == "__main__":
    human_play()
