import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

import trimesh


def view_slices(voxel_matrix, axis=2):
    def get_slice(idx):
        slice_obj = [slice(None), slice(None), slice(None)]
        slice_obj[axis] = idx
        return tuple(slice_obj)

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)

    initial_slice = 0
    img = ax.imshow(voxel_matrix[get_slice(initial_slice)], cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"Slice {initial_slice}")

    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(
        ax_slider,
        "Slice ",
        valmin=0,
        valmax=voxel_matrix.shape[axis] - 1,
        valinit=initial_slice,
        valstep=1,
    )

    def update(val):
        slice_idx = int(slider.val)
        img.set_data(voxel_matrix[get_slice(slice_idx)])
        ax.set_title(f"Slice {slice_idx}")
        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.show()


def view_volume(voxel_matrix: np.ndarray):
    voxel_mesh = trimesh.voxel.VoxelGrid(voxel_matrix)
    voxel_mesh.show(resolution=[800, 800])


if __name__ == "__main__":
    voxel_matrix = np.zeros((32, 32, 32))
    x, y, z = np.ogrid[-16:16, -16:16, -16:16]

    mask = x**2 + y**2 + z**2 <= 16**2
    voxel_matrix[mask] = 1

    view_slices(voxel_matrix)
    # view_volume(voxel_matrix)
