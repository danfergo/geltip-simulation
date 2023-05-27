import time

import numpy as np
import os
import cv2

from sim_model.model import SimulationModel
from sim_model.utils.camera import circle_mask
from sim_model.utils.vis_img import to_panel

fields_size = (120, 160)
sim_size = (640, 480)
# sim_size = (320, 240)
# field = 'linear'
# field = 'geodesic'
# field = 'plane'
field = 'geodesic'
# field = 'planes'
# field = 'geodesic'
# field = 'transport'
# field = 'rtransport'
# field = 'plane2'
rectify_fields = True

__location__ = os.path.dirname(os.path.abspath(__file__))
assets_path = os.path.join(__location__, '../../experimental_setup/geltip/sim_assets/')

light_fields = SimulationModel.load_assets(assets_path, fields_size, sim_size, field, 3)

stack = cv2.resize(cv2.cvtColor(cv2.cvtColor(cv2.imread(assets_path + '/bkg.png'), cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR), sim_size)

model = SimulationModel(**{
    'ia': 0.5,
    'fov': 90,
    'light_sources': [
        {'field': light_fields[1], 'color': [255, 0, 0], 'id': 0.5, 'is': 0.1},  # [108, 82, 255]
        {'field': light_fields[2], 'color': [0, 255, 0], 'id': 0.5, 'is': 0.1},  # [255, 130, 115]
        {'field': light_fields[0], 'color': [0, 0, 255], 'id': 0.5, 'is': 0.1},  # [120, 255, 153]
    ],
    'background_depth': cv2.resize(np.load(assets_path + 'bkg.npy'), sim_size),
    # 'cloud_map': cloud,
    # 'background_img': np.stack([np.zeros(sim_size[::-1]), np.zeros(sim_size[::-1]), np.zeros(sim_size[::-1])], axis=2) * 0.5,
    # 'background_img': (cv2.cvtColor(cv2.imread(assets_path + '/bkg.png'), cv2.COLOR_RGB2BGR) / 255).astype(np.float32),
    'background_img': (stack / 255).astype(np.float32),
    'elastomer_thickness': 0.004,
    'min_depth': 0.026,
    'texture_sigma': 0.00001,
    'elastic_deformation': True,
    'rectify_fields': rectify_fields
})
# ('bkg' if i == 0 else 'depth_' +
depths = [np.load(assets_path + (str(i)) + '.npy') for i in range(6)]
m = circle_mask(sim_size)
m3 = np.stack([m, m, m], axis=2)

# for i, depth in enumerate(depths):
# frame = cv2.resize(depths[0], sim_size)
# elapsed_time = 0
# for i in range(30):
#     start_time = time.time()
#     sim_frame = model.generate(frame)
#     end_time = time.time()
#     elapsed_time += (end_time - start_time)
# print(elapsed_time/30)
# cv2.imshow('frame', sim_frame)
# cv2.waitKey(-1)

tactile_rgb = [
    cv2.cvtColor(model.generate(cv2.resize(depth, sim_size)), cv2.COLOR_RGB2BGR)
    for i, depth in enumerate(depths)
    # if i > 4
]

cv2.imshow('tactile_rgb', to_panel(tactile_rgb, shape=(1, 6)))
cv2.waitKey(-1)

# tactile_rgb = [
#     np.maximum(np.zeros(frame.shape), frame.astype(np.float32) - tactile_rgb[0].astype(np.float32)).astype(np.uint8)
#     for frame in tactile_rgb
# ]
#
# for frame in tactile_rgb:
#     print('imshow', frame.shape, frame.dtype, frame.min(), frame.max())
#
# # show_panel(tactile_rgb)
# cv2.imshow('tactile_rgb', to_panel(tactile_rgb, shape=(2, 3)))
# cv2.waitKey(-1)
# cv2.imshow('depth', to_panel([to_normed_rgb(depth) for depth in depths]))

# for i, depth in enumerate(samples):
# depth = depth.squeeze()
#
#     pts = get_pts_map_from_depth(cam_matrix, depth)
#     tactile_rgb = approach.generate(depth, pts)
#
#     # print(o3d_cloud_raw_pts.shape)
#     # print(self.raw_depth.shape, o3d_cloud_raw_pts.shape[0], 480 * 640)
#
#     # print('--> ', o3d_cloud_raw_pts.shape[0])
#     # if o3d_cloud_raw_pts.shape[0] != 480 * 640:
#     #     return self.raw_depth
#     # print('NO SKIP!')
#
#     # norms = np.sqrt(depth_pts[:, :, 0] ** 2 + depth_pts[:, :, 1] ** 2 + depth_pts[:, :, 2] ** 2)
#     # print(norms.max(), norms.min())
#     #
#     # normalized = depth_pts / np.stack([norms, norms, norms], axis=2)
#     # normalized_norms = np.sqrt(normalized[:, :, 0] ** 2 + normalized[:, :, 1] ** 2 + normalized[:, :, 2] ** 2)
#     # print(normalized_norms.max(), normalized_norms.min())
#
#     # if o3d_cloud_raw_pts.shape[0] == 480 * 640:
#     # print(depth.min(), depth.max())
#     # print(depth.shape)
#     #
#
#     print(tactile_rgb.min(), tactile_rgb.max())
#
#     depth -= depth.min()
#     depth /= depth.max()
#
#     cv2.imshow('depth', depth)
#     cv2.imshow('rgb', tactile_rgb)
#     cv2.waitKey(-1)
#
#     if i >= 3:
#         break
