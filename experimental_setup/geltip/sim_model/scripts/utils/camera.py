import math
import open3d as o3d

import cv2
import numpy as np


def circle_mask(size=(64, 48), border=0):
    """
        used to filter center circular area of a given image,
        corresponding to the geltip surface area
    """
    m = np.zeros((size[1], size[0]))
    m_center = (size[0] // 2, size[1] // 2)
    m_radius = min(size[0], size[1]) // 2 - border
    m = cv2.circle(m, m_center, m_radius, 255, -1)
    m /= 255
    return m.astype(np.float32)


def get_camera_matrix(img_size, fov_deg=70):
    img_width, img_height = img_size

    fov = math.radians(fov_deg)
    f = img_height / (2 * math.tan(fov / 2))
    cx = (img_width - 1) / 2
    cy = (img_height - 1) / 2

    return o3d.camera.PinholeCameraIntrinsic(img_width, img_height, f, f, cx, cy)


def get_cloud_from_depth(cam_matrix, depth):
    o3d_depth = o3d.geometry.Image(depth)
    o3d_cloud = o3d.geometry.PointCloud.create_from_depth_image(o3d_depth, cam_matrix)
    return o3d_cloud
