import os
from math import pi, sin, cos

import numpy as np

from trimesh.exchange.load import load
from experimental_setup.geltip.sim_model.scripts.utils.vis import show_field

assets_path = os.path.dirname(os.path.abspath(__file__)) + '/../assets/'

# cloud_size = (32, 24)
# cloud_size = (64, 48)
cloud_size = (160, 120)

method = 'linear'
prefix = str(cloud_size[0]) + 'x' + str(cloud_size[1])
cloud = np.load(assets_path + '/' + prefix + '_ref_cloud.npy')

mesh = load(assets_path + '/' + prefix + '_aligned_mesh_voxel_e-6.stl', 'stl', force='mesh')
scale = 0.00105
z_trans = -0.015
led_radius = 12
leds = [np.array([cos(a * (2 * pi / 3)) * scale * led_radius, sin(a * (2 * pi / 3)) * scale * led_radius, z_trans]) for a in range(3)]
leds_colors = ['red', 'green', 'blue']

for l in range(len(leds)):
    field = np.load(assets_path + '/' + method + '_' + prefix + '_field_' + str(l) + '.npy')
    cloud_map = cloud.reshape((cloud_size[1], cloud_size[0], 3))

    show_field(cloud_map, field, leds_colors[l], mesh, leds[l], subsample=25)
