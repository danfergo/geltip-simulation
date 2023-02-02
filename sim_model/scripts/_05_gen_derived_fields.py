import os
from math import sqrt

import cv2
import numpy as np

from sim_model import normalize_vectors, derivative, SimulationModel
from sim_model import get_cloud_from_depth, get_camera_matrix

assets_path = os.path.dirname(os.path.abspath(__file__)) + '/../assets/'


def norm(v):
    return sqrt(sum([c ** 2 for c in v]))


def normalize(x):
    norm_ = norm(x)
    return x / norm_ if norm_ != 0 else x


def proj(u, n):
    norm_ = norm(n)
    if norm_ == 0:
        return 0 * u
    return u.dot_vectors(n) * normalize(n)


# project u onto plane with normal n.
# u - proj(u, n)

def gen_rectified_field(base_method, cloud_map, field_i):
    prefix = str(cloud_map.shape[1]) + 'x' + str(cloud_map.shape[0])
    field = np.zeros(cloud_map.shape)

    dx = normalize_vectors(derivative(cloud_map, 'x'))
    dy = normalize_vectors(derivative(cloud_map, 'y'))
    normals = -np.cross(dx, dy)

    plane = np.load(f'{assets_path}/{base_method}_{prefix}_field_{field_i}.npy')
    for i in range(cloud_map.shape[0]):
        for j in range(cloud_map.shape[1]):
            # if log_progress and j == (cloud_map.shape[1] - 1):
            #     progress = ((i * cloud_map.shape[1] + j) / (cloud_map.shape[0] * cloud_map.shape[1]))
            #     print('progress... ' + str(round(progress * 100, 2)) + '%')

            # if m[i, j] > 0.5:
            # lm = cloud_map[i, j] - source_pos
            # d = norm(lm)
            # lm /= (d * d * 50)
            # print(norm(plane[i, j]), norm(normals[i, j]))
            field[i, j] = normalize(plane[i, j]) + 0.1 * normals[i, j]
    return field


def main():
    cloud_size = (160, 120)
    methods = [
        'linear',
        'plane',
        'geodesic',
        'transport'
    ]

    n_leds = 3
    prefix = str(cloud_size[0]) + 'x' + str(cloud_size[1])
    cloud = np.load(assets_path + '/' + prefix + '_ref_cloud.npy')
    cloud_map = cloud.reshape((cloud_size[1], cloud_size[0], 3))

    for method in methods:
        for l in range(n_leds):
            r_field = gen_rectified_field(
                base_method=method,
                cloud_map=cloud_map,
                field_i=l
            )
            np.save(f'{assets_path}/r{method}_{prefix}_field_{l}.npy', r_field)


if __name__ == '__main__':
    main()
