import numpy as np
import matplotlib.pyplot as plt

import cv2


# def flatten(l):
#     return [item for sublist in l for item in sublist]

def show_normalized_img(name, img):
    draw = img.copy()
    draw -= np.min(draw)
    draw = draw / np.max(draw)
    cv2.imshow(name, draw)
    return draw


def to_normed_rgb(depth):
    d = depth - np.min(depth)
    if np.max(d) == 0:
        return np.stack([d, d, d], axis=2)

    d /= np.max(d)
    d *= 255
    d[d < 0] = 0
    d[d > 255] = 255
    d = d.astype(np.uint8)
    return np.stack([d, d, d], axis=2)


def to_panel(frames, shape=(2, 2)):
    return np.concatenate([
        np.concatenate([
            frames[i * shape[1] + j]
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
