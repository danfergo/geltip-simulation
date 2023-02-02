import os

import open3d as o3d
import cv2
import numpy as np

from sim_model \
    import get_camera_matrix, get_cloud_from_depth, circle_mask

base = os.path.dirname(os.path.abspath(__file__)) + '/../assets/'

depth = np.load(base + '/bkg.npy')

# cloud_size = (32, 24)
# cloud_size = (64, 48)
# cloud_size = (320, 240)
cloud_size = (160, 120)
# cloud_size = (640, 480)
# cloud_size = (160, 120)
fov = 90

depth = cv2.resize(depth, cloud_size)

cam_matrix = get_camera_matrix(cloud_size, fov_deg=fov)

o3d_cloud = get_cloud_from_depth(cam_matrix, depth)
o3d_cloud_raw_pts = np.array(o3d_cloud.points)

o3d_cloud_clean = get_cloud_from_depth(cam_matrix, circle_mask(cloud_size) * depth)
o3d_cloud_clean_pts = np.array(o3d_cloud_clean.points)

print(len(o3d_cloud_raw_pts), len(o3d_cloud_clean_pts))

prefix = str(cloud_size[0]) + 'x' + str(cloud_size[1])
np.save(base + '/' + prefix + '_camera_matrix.npy', cam_matrix.intrinsic_matrix)
np.save(base + '/' + prefix + '_ref_cloud.npy', o3d_cloud_raw_pts)
np.save(base + '/' + prefix + '_ref_cloud_clean.npy', o3d_cloud_clean_pts)

o3d.visualization.draw_geometries([o3d_cloud_clean])
