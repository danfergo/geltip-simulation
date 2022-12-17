import cv2
import numpy as np


def color_map(size, color):
    image = np.zeros((size[0], size[1], 3), np.uint8)
    image[:] = color
    return image


def normalize(img):
    if img.dtype == np.uint8:
        return img

    img1 = img - np.min(img)
    mx = np.max(img1)
    if mx < 1e-4:
        return img1
    return img1 / mx


def denormalize(img):
    assert img.dtype == np.float32 or img.dtype == np.float64
    return (img * 255).astype(np.uint8)
