import trimesh
from trimesh.creation import icosphere
from trimesh.scene.scene import Scene

from typing import List, Tuple
from tqdm import tqdm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import io
import os

from transforms import apply_rotation


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
    beam = trimesh.creation.cylinder(radius=radius, height=length, sections=8)

    lungs_position = lungs_mesh.bounding_box.centroid
    translation = position + lungs_position

    direction = direction / np.linalg.norm(direction)
    z_axis = np.array([0, 0, 1])
    angle = trimesh.transformations.angle_between_vectors(z_axis, direction)
    cross = np.cross(z_axis, direction)

    if np.linalg.norm(cross) > 1e-6:
        rotation = trimesh.transformations.rotation_matrix(angle, cross)
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


def get_tumour(position, radius):
    tumour = icosphere(radius=radius, subdivisions=2)
    tumour.apply_translation(position)

    set_color_tumour(tumour)
    return tumour


def generate_tumour(
    lungs_mesh: trimesh.Trimesh,
    mean_radius: float = 0.1,
    std_radius: float = 0.05,
    resolution: int = 20,
    min_size: float = 0.05,
) -> List[trimesh.Trimesh]:
    bounds = lungs_mesh.bounds
    min_bound, max_bound = bounds[0], bounds[1]

    valid_tumour = False
    while not valid_tumour:
        position, radius = get_random_sphere_bounded(
            mean_radius, std_radius, min_bound, max_bound
        )
        if radius < min_size:
            continue

        valid_tumour = is_inside(lungs_mesh, resolution, position, radius)

    tumour = get_tumour(position, radius)

    return tumour, (position, radius)


def get_random_sphere_bounded(
    mean_radius, std_radius, min_bound, max_bound, accuracy=2
):
    position = np.round(np.random.uniform(min_bound, max_bound), accuracy)
    radius = np.round(np.abs(np.random.normal(mean_radius, std_radius)), accuracy)
    return position, radius


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
    # tumour filename format: x_y_z_radius.npy

    lungs_save_path = os.path.join(save_path, "lungs.npy")
    tumours_save_dir = os.path.join(save_path, "tumours")
    os.makedirs(tumours_save_dir, exist_ok=True)

    lungs = load_lungs_model()
    lungs_shape, lungs_bounds, lungs_matrix = voxelize(lungs, pitch)
    np.save(lungs_save_path, lungs_matrix)

    for _ in tqdm(range(n_tumours)):

        tumour, (position, radius) = generate_tumour(lungs)
        tumour_matrix = embed_tumour_in_lungs(tumour, lungs_bounds, lungs_matrix, pitch)

        tumour_filename = f"{position[0]}_{position[1]}_{position[2]}_{radius}.npy"
        tumour_save_path = os.path.join(tumours_save_dir, tumour_filename)
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


def export_transformation_matrix():
    human_model = load_human_model()
    lungs = load_lungs_model()
    scene = Scene([lungs, human_model])
    scene.show(resolution=(800, 600))
    camera_transform = scene.camera_transform

    print(camera_transform)
    np.save("camera_transform.npy", camera_transform)


def export_scene(scene, resolution=(800, 600)):
    image_data = scene.save_image(resolution=resolution, visible=True)
    image = Image.open(io.BytesIO(image_data))
    image = image.convert("RGBA")
    return image


def create_animation(
    tumours_data,
    beams_data,
    filename="animations/test.gif",
    export_gif=True,
    window=False,
):
    human_model = load_human_model()
    lungs = load_lungs_model()

    tumours = [get_tumour(position, radius) for (position, radius) in tumours_data]
    beams = [
        create_beam(lungs, position, direction) for (position, direction) in beams_data
    ]

    camera_transform = np.load("camera_transform.npy")

    frames = []
    for i in range(len(beams)):

        scene = Scene([tumours] + beams[: i + 1] + [lungs, human_model])
        scene.camera_transform = camera_transform

        image = export_scene(scene)
        frames.append(image)

    if export_gif:
        frames[0].save(
            filename,
            save_all=True,
            append_images=frames[1:],
            duration=500,
            loop=0,
        )

    if window:
        fig, ax = plt.subplots(figsize=(8, 6))
        img = ax.imshow(frames[0])
        ax.axis("off")

        def update(frame):
            img.set_data(frame)
            return (img,)

        ani = FuncAnimation(fig, update, frames=frames, interval=500, blit=True)
        plt.show()


def test_scene():
    human_model = load_human_model()
    lungs = load_lungs_model()

    tumour, _ = generate_tumour(lungs)

    beam = create_beam(lungs, np.array([0, 0, 0]), np.array([0, 0, 1]))

    beams = [beam]

    scene = Scene([tumour] + beams + [lungs, human_model])
    scene.show(resolution=(800, 600))


def generate_rotation_sequence(start_direction, rotation, n_steps):
    rotation_vector = np.array(rotation) / n_steps

    direction = start_direction
    sequence = [direction]
    for _ in range(n_steps):
        direction, overshoot = apply_rotation(
            direction, rotation_vector, min_angle=np.pi / 4
        )
        sequence.append(direction)

    return sequence


def test_animation():
    lungs = load_lungs_model()

    _, (position, radius) = generate_tumour(lungs)
    tumours_data = [(position, radius)]

    beam_position = np.array([0, 0, 0])
    beam_rotations = generate_rotation_sequence(
        np.array([0, 1, 0]), np.array([0, 0, -np.pi / 4]), 10
    )
    beams_data = [(beam_position, rotation) for rotation in beam_rotations]

    create_animation(tumours_data, beams_data, export_gif=True, window=True)


if __name__ == "__main__":
    test_animation()
