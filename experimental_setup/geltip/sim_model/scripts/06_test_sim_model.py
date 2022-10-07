import numpy as np
import os
import cv2

from experimental_setup.geltip.sim_model.model import SimulationModel
from experimental_setup.geltip.sim_model.scripts.utils.vis import show_panel

fields_size = (160, 120)
sim_size = (640, 480)
field = 'linear'
assets_path = os.path.dirname(os.path.abspath(__file__)) + '/../assets/'

cloud, light_fields = SimulationModel.load_assets(assets_path, fields_size, sim_size, field, 3)

model = SimulationModel(**{
    'ia': 0.8,
    'light_sources': [
        {'field': light_fields[0], 'color': [0, 0, 255], 'id': 0.5, 'is': 0.2},  # [108, 82, 255]
        {'field': light_fields[1], 'color': [255, 0, 0], 'id': 0.5, 'is': 0.2},  # [255, 130, 115]
        {'field': light_fields[2], 'color': [0, 255, 0], 'id': 0.5, 'is': 0.2},  # [120, 255, 153]
    ],
    'background_depth': np.load(assets_path + 'bkg.npy'),
    'cloud_map': cloud,
    'background_img': np.ones(sim_size[::-1] + (3,), dtype=np.float32) * 255.0,  # cv2.imread(base + '/bkg.jpg'),
    'elastomer_thickness': 0.004,
    'min_depth': 0.026,
    'texture_sigma': 0.00001,
    'elastic_deformation': True
})

depths = [np.load(assets_path + ('bkg' if i == 0 else 'depth_' + str(i)) + '.npy') for i in range(4)]

tactile_rgb = [model.generate(cv2.resize(depth, sim_size)) for i, depth in enumerate(depths) if i != 0]

show_panel(tactile_rgb)
# cv2.imshow('tactile_rgb', to_panel(tactile_rgb))
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
