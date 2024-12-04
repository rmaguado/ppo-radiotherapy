import trimesh
from trimesh.creation import icosphere
from trimesh.voxel.base import VoxelGrid
from trimesh.scene.scene import Scene

from typing import List
import numpy as np
import os


def set_color_body(mesh, opacity=0.25) -> None:
    opacity_8bit = int(255 * opacity)
    color = np.array([0, 0, 0, opacity_8bit])

    mesh.visual.face_colors[:] = color
    mesh.visual.vertex_colors[:] = color


def set_color_lungs(mesh, opacity=0.5) -> None:
    opacity_8bit = int(255 * opacity)
    color = np.array([255, 0, 0, opacity_8bit])

    mesh.visual.face_colors[:] = color
    mesh.visual.vertex_colors[:] = color


def set_color_tumour(mesh, opacity=1.0) -> None:
    opacity_8bit = int(255 * opacity)
    color = np.array([0, 255, 0, opacity_8bit])

    mesh.visual.face_colors[:] = color
    mesh.visual.vertex_colors[:] = color


def set_color_beam(mesh, opacity=0.7) -> None:
    opacity_8bit = int(255 * opacity)
    color = np.array([0, 0, 255, opacity_8bit])  # Blue for the beam
    mesh.visual.face_colors[:] = color
    mesh.visual.vertex_colors[:] = color


def load_lungs_model() -> trimesh.Trimesh:
    lungs_model = trimesh.load("./models/downsampled/lungs.obj")

    lungs_model.apply_transform(trimesh.transformations.scale_matrix(0.0135))

    lungs_model.apply_transform(
        trimesh.transformations.rotation_matrix(-np.pi, (1, 0, 0))
    )
    lungs_model.apply_transform(
        trimesh.transformations.rotation_matrix(np.pi / 2, (0, 1, 0))
    )

    lungs_model.apply_translation([-14.8, 0.12, 0.2])

    set_color_lungs(lungs_model)
    return lungs_model


def load_human_model() -> trimesh.Trimesh:
    human_model = trimesh.load("./models/downsampled/man.obj")
    human_model.apply_transform(
        trimesh.transformations.rotation_matrix(-np.pi / 2, (1, 0, 0))
    )
    human_model.apply_transform(
        trimesh.transformations.rotation_matrix(np.pi / 2, (0, 1, 0))
    )
    set_color_body(human_model)
    return human_model


def create_beam(
    lungs_mesh: trimesh.Trimesh,
    position: np.ndarray,
    direction: np.ndarray,
    radius: float = 0.1,
    length: float = 10.0,
) -> trimesh.Trimesh:

    lungs_position = lungs_mesh.bounding_box.centroid
    translation = position + lungs_position

    direction = direction / np.linalg.norm(direction)
    z_axis = np.array([0, 0, 1])
    rotation = trimesh.transformations.rotation_matrix(
        trimesh.transformations.angle_between_vectors(z_axis, direction),
        np.cross(z_axis, direction),
    )

    beam = trimesh.creation.cylinder(radius=radius, height=length, sections=8)
    beam.apply_transform(rotation)
    beam.apply_translation(translation)
    set_color_beam(beam)

    return beam


def is_inside(
    lungs_mesh: trimesh.Trimesh, resolution: int, position: np.ndarray, radius: float
) -> bool:
    valid_tumour = True
    for _ in range(resolution):
        direction = np.random.normal(size=3)
        direction /= np.linalg.norm(direction)
        surface_point = position + direction * radius

        if not lungs_mesh.contains([surface_point]):
            valid_tumour = False
            break
    return valid_tumour


def generate_tumour(
    lungs_mesh: trimesh.Trimesh,
    mean_radius: float = 0.1,
    std_radius: float = 0.05,
    resolution: int = 20,
) -> List[trimesh.Trimesh]:
    bounds = lungs_mesh.bounds
    min_bound, max_bound = bounds[0], bounds[1]

    valid_tumour = False
    while not valid_tumour:
        position = np.random.uniform(min_bound, max_bound)
        radius = np.abs(np.random.normal(mean_radius, std_radius))

        valid_tumour = is_inside(lungs_mesh, resolution, position, radius)
    tumour = icosphere(radius=radius, subdivisions=2)
    tumour.apply_translation(position)

    set_color_tumour(tumour)

    return tumour, (position, radius)


def embed_tumour_in_lungs(tumour, lungs_bounds, lungs_matrix, pitch):
    tumour_shape, tumour_bounds, tumour_matrix = voxelize(tumour, pitch)

    tumour_matrix_full = np.zeros_like(lungs_matrix).astype(np.float32)

    offset = ((tumour_bounds[0] - lungs_bounds[0]) / pitch).astype(int)

    tumour_matrix_full[
        offset[0] : offset[0] + tumour_shape[0],
        offset[1] : offset[1] + tumour_shape[1],
        offset[2] : offset[2] + tumour_shape[2],
    ] = tumour_matrix

    return tumour_matrix_full


def voxelize(mesh, pitch):
    voxelized_mesh = mesh.voxelized(pitch=pitch, method="subdivide").fill()
    shape = voxelized_mesh.shape
    bounds = voxelized_mesh.bounds
    matrix = voxelized_mesh.matrix
    return shape, bounds, matrix


def pregenerate_voxel_data(save_path: str, n_tumours: int, pitch=0.05) -> None:
    lungs_save_path = os.path.join(save_path, "lungs.npy")
    tumours_save_dir = os.path.join(save_path, "tumours")
    os.makedirs(tumours_save_dir, exist_ok=True)

    lungs = load_lungs_model()
    lungs_shape, lungs_bounds, lungs_matrix = voxelize(lungs, pitch)
    np.save(lungs_save_path, lungs_matrix)

    for tumour_idx in range(n_tumours):

        tumour, _ = generate_tumour(lungs)
        tumour_matrix = embed_tumour_in_lungs(tumour, lungs_bounds, lungs_matrix, pitch)

        tumour_save_path = os.path.join(tumours_save_dir, f"{tumour_idx:04d}.npy")
        np.save(tumour_save_path, tumour_matrix)


def beam_voxels(base_matrix, position, direction):
    """
    Discretizes a line in a 3D matrix using Xiaolin Wu's Algorithm

    Parameters:
        base_matrix (np.ndarray): The 3D matrix defining the space.
        position (tuple): The starting position of the beam as a 3D tuple (x, y, z).
        direction (tuple): The direction vector of the beam.

    Returns:
        np.ndarray: The 3D matrix with the discretized beam applied.
    """
    output = np.zeros_like(base_matrix, dtype=np.float32)

    position = np.array(position, dtype=np.float32)
    direction = np.array(direction, dtype=np.float32)
    direction /= np.linalg.norm(direction)

    grid_size = base_matrix.shape

    t_min = []
    t_max = []
    for i in range(3):
        if direction[i] != 0:
            t1 = (-position[i]) / direction[i]
            t2 = (grid_size[i] - 1 - position[i]) / direction[i]
            t_entry_i = min(t1, t2)
            t_exit_i = max(t1, t2)
        else:
            if position[i] < 0 or position[i] > grid_size[i] - 1:
                return output
            else:
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

    if direction[dominant_axis] > 0:
        start_voxel = int(
            np.floor(position[dominant_axis] + t_entry * direction[dominant_axis])
        )
        end_voxel = int(
            np.floor(position[dominant_axis] + t_exit * direction[dominant_axis])
        )
        step = 1
    else:
        start_voxel = int(
            np.floor(position[dominant_axis] + t_entry * direction[dominant_axis])
        )
        end_voxel = int(
            np.floor(position[dominant_axis] + t_exit * direction[dominant_axis])
        )
        step = -1

    intery = position[other_axes[0]] + t_entry * direction[other_axes[0]]
    interz = position[other_axes[1]] + t_entry * direction[other_axes[1]]

    gradient_y = direction[other_axes[0]] / direction[dominant_axis]
    gradient_z = direction[other_axes[1]] / direction[dominant_axis]

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


def test_add_beam():
    import matplotlib.pyplot as plt

    lungs = load_lungs_model()
    lungs_shape, lungs_bounds, lungs_matrix = voxelize(lungs, pitch=0.05)

    test_position = np.array([0, 0, 0])
    test_direction = np.array([0, 1, 0.75])

    dose_matrix = beam_voxels(lungs_matrix, test_position, test_direction)

    plt.imshow(dose_matrix[0, :, :])
    plt.show()


def test_scene():
    human_model = load_human_model()
    lungs = load_lungs_model()

    tumour, _ = generate_tumour(lungs)

    beam = create_beam(lungs, np.array([0, 0, 0]), np.array([0, 1, 0]))

    beams = [beam]

    scene = Scene([tumour] + beams + [lungs, human_model])
    scene.show(resolution=(800, 600))


if __name__ == "__main__":
    test_add_beam()
