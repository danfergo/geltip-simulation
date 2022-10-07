import cv2
import numpy as np
from trimesh import Scene

from trimesh.primitives import Sphere
from trimesh.creation import cone, cylinder
from trimesh.geometry import align_vectors
from trimesh.transformations import translation_matrix

# images visualization
from experimental_setup.geltip.sim_model.scripts.utils.camera import circle_mask

import matplotlib.pyplot as plt


def to_normed_rgb(depth):
    depth -= np.min(depth)
    if np.max(depth) == 0:
        return depth

    depth /= np.max(depth)
    depth *= 255
    depth[depth < 0] = 0
    depth[depth > 255] = 255
    depth = depth.astype(np.uint8)
    return np.stack([depth, depth, depth], axis=2)


def to_panel(frames, shape=(2, 2)):
    return np.concatenate([
        np.concatenate([
            frames[i * shape[0] + j]
            for j in range(shape[1])
        ], axis=1)
        for i in range(shape[0])
    ], axis=0)


def show_panel(frames, shape=None):
    if shape is None:
        shape = (1, len(frames))

    fig, axes = plt.subplots(*shape)
    fig.set_size_inches(shape[1] * 10, shape[0] * 10)

    axes = axes if hasattr(axes, "__len__") else [axes]
    axes = axes if hasattr(axes[0], "__len__") else [axes]

    for i in range(shape[0]):
        for j in range(shape[1]):
            axes[i][j].imshow(frames[i * shape[1] + j])

    plt.show()


# (tri)meshes/3d visualization
colors = {
    'green': [0, 255, 0, 255],
    'blue': [0, 0, 255, 255],
    'red': [255, 0, 0, 255],
    'pink': [255, 0, 255, 255],
    'yellow': [255, 255, 0, 255],
    'black': [0, 0, 0, 255],
}


def sphere(p, color='green'):
    s = Sphere(radius=0.002, center=p, subdivisions=1)
    s.visual.face_colors = colors[color]
    return s


def arrow(p, v, color='red', scale=0.25, length=0.25):
    r = align_vectors(
        np.array([0, 0, 1]),
        v,
        return_angle=False
    )

    trans = translation_matrix(p)
    transform = np.matmul(trans, r)
    h = cone(height=0.004 * length, radius=0.002 * scale, transform=transform)
    h.visual.face_colors = colors[color]

    t = cylinder(height=0.004 * scale, radius=0.001 * scale, transform=transform)

    t.visual.face_colors = [0, 0, 0, 255]

    return [h, t]


def show_field(cloud_map, field=None, field_color=None, mesh=None, source_pt=None, subsample=25):
    m = circle_mask((cloud_map.shape[1], cloud_map.shape[0]))
    arrows = []
    if field is not None:
        for i in range(cloud_map.shape[0]):
            for j in range(cloud_map.shape[1]):
                if m[i, j] > 0.5 and (not subsample or (i * cloud_map.shape[1] + j) % subsample == 0):
                    arrows.extend(arrow(cloud_map[i, j], field[i, j], color=field_color))

    source_pt_sphere = sphere(source_pt, 'black') if source_pt is not None else None

    if mesh is not None:
        mesh.visual.face_colors = [127, 127, 127, 127]

    scene = Scene(([mesh] if mesh else [])
                  + ([source_pt_sphere] if source_pt_sphere else [])
                  + [*arrows])
    scene.show()


def plot_depth_lines(depth_clouds, depth, row=None,
                     colors=['red', 'green', 'blue'],
                     legends=[None, None, None]):
    if row is None:
        row = depth_clouds[0].shape[0] // 2

    fig, (ax1, ax2) = plt.subplots(1, 2)

    for i, depth_cloud in enumerate(depth_clouds):
        xs = depth_cloud[row, :, 0]
        ys = depth_cloud[row, :, 2]

        ax1.plot(xs,
                 ys,
                 color=colors[i],
                 label=legends[i])

    ax1.axis('equal')
    ax1.legend()

    depth_rgb = to_normed_rgb(depth)
    depth_rgb = cv2.line(depth_rgb, (0, row), (depth_rgb.shape[1], row), (255, 0, 0), 1)

    ax2.imshow(depth_rgb)
    plt.show()
