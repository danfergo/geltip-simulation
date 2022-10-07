from yarok import ConfigBlock
from yarok.components_manager import component
from yarok.mjc.interface import InterfaceMJC

import numpy as np
import cv2
import os
import open3d as o3d

import math

from time import time


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


class GelTipInterfaceMJC:

    def __init__(self, interface: InterfaceMJC):
        self.interface = interface
        self.raw_depth = np.zeros((480, 640))
        self.tactile_rgb = np.zeros((480, 640))
        self.last_update = 0

        base = os.path.dirname(__file__) + '/simulation_model'

        cloud_size = (160, 120)
        method = 'geo'
        prefix = str(cloud_size[0]) + 'x' + str(cloud_size[1])

        return

        cloud = np.load(base + '/fields/' + prefix + '_ref_cloud.npy')
        cloud = cloud.reshape((cloud_size[1], cloud_size[0], 3))
        cloud = cv2.resize(cloud, (640, 480))

        normals = np.load(base + '/fields/' + prefix + '_surface_normals.npy')
        normals = normals.reshape((cloud_size[1], cloud_size[0], 3))
        normals = cv2.resize(normals, (640, 480))

        light_fields = [
            normalize_vectors(
                cv2.GaussianBlur(
                    cv2.resize(np.load(base + '/fields/' + method + '_' + prefix + '_field_ ' + str(l) + '.npy')
                               .astype(np.float64), (640, 480))
                    , (15, 15), cv2.BORDER_DEFAULT)
            )
            for l in range(3)
        ]

        self.approach = SimulationApproach(**{
            'light_sources': [
                {'field': light_fields[0], 'color': [108, 82, 255], 'kd': 0.1, 'ks': 0.9},
                {'field': light_fields[1], 'color': [255, 130, 115], 'kd': 0.1, 'ks': 0.9},
                {'field': light_fields[2], 'color': [120, 255, 153], 'kd': 0.1, 'ks': 0.9},
            ],
            'cloud_map': cloud,
            'normals': normals,
            'background_img': cv2.imread(base + '/bkg.jpg'),
            'ka': 0.8,
            'px2m_ratio': 5.4347826087e-05,
            'elastomer_thickness': 0.004,
            'min_depth': 0.026,
            'texture_sigma': 0.000002
        })

        self.cam_matrix = get_camera_matrix((640, 480))

    def _read(self):
        rgb, depth = self.interface.read_camera('camera', (480, 640), depth=True, rgb=True)
        self.raw_rgb = rgb
        self.raw_depth = depth

    def read(self):
        self._read()

        t = time()
        if self.last_update > t - 0.1:
            return self.raw_depth

        self.last_update = t
        self.raw_rgb = rgb
        self.raw_depth = depth

        # print(self.raw_depth.shape)
        # self.raw_depth = frame[1]
        return self.raw_depth

        o3d_cloud = get_cloud_from_depth(self.cam_matrix, self.raw_depth * 10)
        o3d_cloud_raw_pts = np.asarray(o3d_cloud.points)
        # print(o3d_cloud_raw_pts.shape)
        # print(self.raw_depth.shape, o3d_cloud_raw_pts.shape[0], 480 * 640)

        # print('--> ', o3d_cloud_raw_pts.shape[0])
        if o3d_cloud_raw_pts.shape[0] != 480 * 640:
            print('SKIP!')
            return self.raw_depth

        depth = o3d_cloud_raw_pts.reshape((480, 640, 3)) / 10

        # if o3d_cloud_raw_pts.shape[0] == 480 * 640:
        self.tactile_rgb = self.approach.generate(depth)

        # self.tactile_rgb = frame[0]
        return self.tactile_rgb

    def read_depth(self):
        self._read()
        return self.raw_depth


class GelTipInterfaceHW:

    def __init__(self, config: ConfigBlock):
        self.cap = cv2.VideoCapture(config['cam_id'])
        if not self.cap.isOpened():
            raise Exception('GelTip cam ' + str(config['cam_id']) + ' not found')

        self.fake_depth = np.zeros((640, 480), np.float32)

    def read(self):
        [self.cap.read() for _ in range(10)]  # skip frames in the buffer.
        ret, frame = self.cap.read()
        return frame

    def read_depth(self):
        return self.fake_depth


@component(
    tag="geltip",
    interface_mjc=GelTipInterfaceMJC,
    interface_hw=GelTipInterfaceHW,
    probe=lambda c: {'camera': c.read()},
    # language=xml
    template="""
        <mujoco>
            <asset>
                <material name="glass_material" rgba="1 1 1 0.1"/>
                <material name="white_elastomer" rgba="1 1 1 1"/>
                <material name="black_plastic" rgba=".3 .3 .3 1"/>
                
                <material name="label_color" rgba="${label_color} 1.0"/>
        
                <mesh name="geltip_shell" file="meshes/shell_open.stl" scale="0.001 0.001 0.001"/>
                <mesh name="geltip_sleeve" file="meshes/sleeve_open.stl" scale="0.001 0.001 0.001"/>
                <mesh name="geltip_mount" file="meshes/mount.stl" scale="0.001 0.001 0.001"/>
                
                <!-- the glass -->
                <mesh name="geltip_glass" file="meshes/glass_long.stl" scale="0.00099 0.00099 0.00099"/>
                
                <!-- the outter elastomer, for visual purposes -->
                <mesh name="geltip_elastomer" file="meshes/elastomer_long.stl" scale="0.0011 0.0011 0.0011"/>  
                
                <!-- inverted mesh, for limiting the depth map-->
                <!-- changing this mesh changes the depth maps -->
                <mesh name="geltip_elastomer_inv" file="meshes/elastomer_long_inv.stl" scale="0.00105 0.00105 0.00105"/>
        
            </asset>
            <worldbody>
                <body name="geltip">
                    <geom type="sphere" 
                        density="0.1"
                        material="label_color" 
                        size="0.005" 
                        pos="0.0 0.012 -0.025"/>
                    <geom density="0.1" type="mesh" mesh="geltip_shell" material="black_plastic"/>
                    <geom density="0.1" type="mesh" mesh="geltip_sleeve" material="black_plastic"/>
                    <geom density="0.1" type="mesh" mesh="geltip_mount" material="black_plastic"/>
                    <camera name="camera" pos="0 0 0.01" zaxis="0 0 -1" fovy="70"/>
                    <body>
                    
                       <!-- mesh, to serve as the glass and detect collisions -->
                       <geom density="0.1" type="mesh" 
                              mesh="geltip_glass" 
                              pos="0.0 0.0 -0.003"
                              solimp="1.0 1.2 0.001 0.5 2" 
                              solref="0.02 1"
                              material="glass_material" /> 
                              
                       <!-- inverted, mesh, for limiting the depth-map -->
                       <!-- changing this geom/mesh changes the depth maps -->
                       <!-- 32 for contype and conaffinity disable collisions -->     
                       <geom  type="mesh" 
                              mesh="geltip_elastomer_inv" 
                              contype="32"  
                              conaffinity="32" 
                              pos="0.0 0.0 -0.005"
                              material="white_elastomer" />
                       
                       <!-- white elastomer, for visual purposes -->
                       <geom density="0.1" 
                              type="mesh" 
                              mesh="geltip_elastomer" 
                              friction="1 0.05 0.01" 
                              contype="32" 
                              conaffinity="32" 
                              pos="0.0 0.0 -0.007"
                              material="white_elastomer"/>
                    </body>
        
                </body>
            </worldbody>
        </mujoco>
    """
)
class GelTip:

    def __init__(self):
        """
            Geltip driver as proposed in
            https://danfergo.github.io/geltip-simulation/

            The frame method gets the depth map from the simula
        """
        pass

    def read(self):
        pass

    def read_depth(self):
        pass
