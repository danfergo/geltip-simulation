import os
from math import pi, sin, cos, sqrt

import cv2
import numpy as np

from experimental_setup.geltip.sim_model.model import normalize_vectors, derivative, SimulationModel
from experimental_setup.geltip.sim_model.scripts.utils.camera import get_cloud_from_depth, get_camera_matrix

assets_path = os.path.dirname(os.path.abspath(__file__)) + '/../assets/'

cloud_size = (160, 120)

n_leds = 3
prefix = str(cloud_size[0]) + 'x' + str(cloud_size[1])
cloud = np.load(assets_path + '/' + prefix + '_ref_cloud.npy')

depth_path = "/home/danfergo/Projects/PhD/geltip_simulation/geltip_dataset/dataset/sim_depth/cylinder/bkg.npy"
depth = cv2.resize(np.load(depth_path), cloud_size, interpolation=cv2.INTER_LINEAR)

s = np.array(get_cloud_from_depth(get_camera_matrix(depth.shape[::-1]), depth).points) \
    .reshape((depth.shape[0], depth.shape[1], 3))

s, _ = SimulationModel.load_assets(assets_path, cloud_size, cloud_size, 'geodesic', 3)


dx = normalize_vectors(derivative(s, 'x'))
dy = normalize_vectors(derivative(s, 'y'))
normals = - np.cross(dx, dy)


def norm(v):
    return sqrt(sum([x ** 2 for x in v]))


def proj(u, n):
    norm_ = norm(n)
    if norm_ == 0:
        return 0 * u
    return ((u * n) / (norm_ ** 2)) * n


def main():
    for l in range(n_leds):
        geo_field = np.load(assets_path + '/geodesic_' + prefix + '_field_' + str(l) + '.npy')

        rgeo_field = np.zeros(geo_field.shape)

        for i in range(depth.shape[0]):
            for j in range(depth.shape[1]):
                u = geo_field[i, j]
                n = normals[i, j]

                rgeo_field[i, j] = u - proj(u, n)

        np.save(assets_path + '/rgeodesic_' + prefix + '_field_' + str(l) + '.npy', rgeo_field)


if __name__ == '__main__':
    main()
