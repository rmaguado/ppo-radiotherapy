import trimesh
from trimesh.creation import icosphere
from trimesh.voxel.base import VoxelGrid
from trimesh.scene.scene import Scene

from typing import List
import numpy as np


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


def generate_n_tumours(
    lungs_mesh: trimesh.Trimesh,
    n_tumours: int = 5,
    mean_radius: float = 0.1,
    std_radius: float = 0.05,
    resolution: int = 20,
) -> List[trimesh.Trimesh]:
    """
    Generate `n_tumours` random tumours (nodules) entirely within the lungs mesh.

    Args:
        lungs_mesh (trimesh.Trimesh): The 3D model of the lungs.
        n_tumours (int): The number of tumours to generate.
        mean_radius (float): Mean radius of the tumours.
        std_radius (float): Standard deviation of the radius of the tumours.
        resolution (int): Number of points to sample on the sphere surface for validation.

    Returns:
        List[trimesh.Trimesh]: List of tumour meshes.
    """
    bounds = lungs_mesh.bounds
    min_bound, max_bound = bounds[0], bounds[1]

    tumours = []
    for _ in range(n_tumours):
        valid_tumour = False
        while not valid_tumour:
            position = np.random.uniform(min_bound, max_bound)
            radius = np.abs(np.random.normal(mean_radius, std_radius))

            valid_tumour = is_inside(lungs_mesh, resolution, position, radius)
        tumour = trimesh.primitives.Sphere(radius=radius, center=position)
        set_color_tumour(tumour)

        tumours.append(tumour)

    return tumours


def voxelize(mesh, pitch=0.5) -> VoxelGrid:
    voxelized = mesh.voxelized(pitch=pitch, method="ray")
    return voxelized.matrix


def main():
    human_model = load_human_model()
    lungs = load_lungs_model()

    tumours = generate_n_tumours(lungs, n_tumours=5)

    scene = Scene(tumours + [lungs, human_model])
    scene.show(resolution=(800, 600))


if __name__ == "__main__":
    main()
