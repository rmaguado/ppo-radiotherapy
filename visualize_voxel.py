import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


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


def view_observation_slices(voxel_matrix, axis=2):
    def get_slice(volume, idx):
        slice_obj = [slice(None), slice(None), slice(None)]
        slice_obj[axis] = idx
        return volume[tuple(slice_obj)]

    max_slices = voxel_matrix[0].shape[axis]

    fig, axes = plt.subplots(2, 2, figsize=(6, 8))
    plt.tight_layout()

    initial_slice = 0

    imgs = []
    volumes = [voxel_matrix[i] for i in range(4)]
    for ax, volume in zip(axes.flat, volumes):
        img = ax.imshow(get_slice(volume, initial_slice), cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        imgs.append(img)

    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(
        ax_slider,
        "Slice ",
        valmin=0,
        valmax=max_slices - 1,
        valinit=initial_slice,
        valstep=1,
    )
    plt.subplots_adjust(bottom=0.1)

    def update(val):
        slice_idx = int(slider.val)
        for ax, img, volume in zip(axes.flat, imgs, volumes):
            img.set_data(get_slice(volume, slice_idx))
        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.show()


if __name__ == "__main__":
    voxel_matrix = np.zeros((32, 32, 32))
    x, y, z = np.ogrid[-16:16, -16:16, -16:16]

    mask = x**2 + y**2 + z**2 <= 16**2
    voxel_matrix[mask] = 1

    view_slices(voxel_matrix)
