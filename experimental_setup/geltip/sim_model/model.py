#!/usr/bin/env python
import cv2
import numpy as np

import scipy.ndimage.filters as fi

from experimental_setup.geltip.sim_model.scripts.utils.camera \
    import get_camera_matrix, get_cloud_from_depth
from experimental_setup.geltip.sim_model.scripts.utils.vis import plot_depth_lines, show_field, show_panel


def dot(a, b):
    return np.sum(np.multiply(a, b), axis=2)


def normalize_vectors(m, zero=1.e-9):
    n = np.sqrt(np.sum(np.square(m), axis=2))
    n = np.where(((-1 * zero) < n) & (n < zero), 1, n)
    n = n[:, :, np.newaxis].repeat(3, axis=2)
    return m / n


PKG_PATH = '/'

""" 
    Utils section
"""


def show_normalized_img(name, img):
    draw = img.copy()
    draw -= np.min(draw)
    draw = draw / np.max(draw)
    cv2.imshow(name, draw)
    return draw


def gkern2(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""

    # create nxn zeros
    inp = np.zeros((kernlen, kernlen))
    # set element at the middle to one, a dirac delta
    inp[kernlen // 2, kernlen // 2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    return fi.gaussian_filter(inp, nsig)


def derivative(mat, direction):
    assert (direction == 'x' or direction == 'y'), "The derivative direction must be 'x' or 'y'"
    kernel = None
    if direction == 'x':
        kernel = [[-1.0, 0.0, 1.0]]
    elif direction == 'y':
        kernel = [[-1.0], [0.0], [1.0]]
    kernel = np.array(kernel, dtype=np.float32)
    return cv2.filter2D(mat, -1, kernel) / 2.0


""" 
    GelSight Simulation
"""


class SimulationModel:

    def __init__(self, **config):
        self.default_ks = 0.15
        self.default_kd = 0.5
        self.default_alpha = 100
        self.ia = config['ia'] or 0.8

        self.lights = config['light_sources']

        self.s_ref = config['cloud_map']
        self.bkg_depth = config['background_depth']
        self.background_img = config['background_img']

        self.apply_elastic_deformation = config['elastic_deformation'] if 'elastic_deformation' in config else False
        self.elastomer_thickness = config['elastomer_thickness']
        self.min_depth = config['min_depth']

        # pre compute & defaults
        self.ambient = config['background_img']

        for light in self.lights:
            light['ks'] = light['ks'] if 'ks' in light else self.default_ks
            light['kd'] = light['kd'] if 'kd' in light else self.default_kd
            light['alpha'] = light['alpha'] if 'alpha' in light else self.default_alpha

            light['color_map'] = np.tile(np.array(np.array(light['color']) / 255.0)
                                         .reshape((1, 1, 3)), self.s_ref.shape[0:2] + (1,))

        self.texture_sigma = config['texture_sigma'] or 0.00001
        self.t = config['t'] if 't' in config else 3
        self.sigma = config['sigma'] if 'sigma' in config else 7
        self.kernel_size = config['sigma'] if 'sigma' in config else 21
        self.max_depth = self.min_depth + self.elastomer_thickness

    @staticmethod
    def load_assets(assets_path, input_res, output_res, lf_method, n_light_sources):
        prefix = str(input_res[0]) + 'x' + str(input_res[1])

        cloud = np.load(assets_path + '/' + prefix + '_ref_cloud.npy')
        cloud = cloud.reshape((input_res[1], input_res[0], 3))
        cloud = cv2.resize(cloud, output_res)

        # normals = np.load(assets_path + '/' + prefix + '_surface_normals.npy')
        # normals = normals.reshape((input_res[1], input_res[0], 3))
        # normals = cv2.resize(normals, output_res)

        light_fields = [
            normalize_vectors(
                cv2.resize(
                    cv2.GaussianBlur(
                        # cv2.resize(
                            np.load(assets_path + '/' + lf_method + '_' + prefix + '_field_' + str(l) + '.npy'),
                            # (80, 60), interpolation=cv2.INTER_LINEAR),
                        (25, 25), 0),
                    output_res, interpolation=cv2.INTER_LINEAR)
            )
            for l in range(n_light_sources)
        ]
        # normals,
        return cloud, light_fields

    def protrusion_map(self, original, not_in_touch):
        protrusion_map = np.copy(original)
        protrusion_map[not_in_touch >= self.max_depth] = self.max_depth
        return protrusion_map

    def segments(self, depth_map):
        not_in_touch = np.copy(depth_map)
        not_in_touch[not_in_touch < self.max_depth] = 0.0
        not_in_touch[not_in_touch >= self.max_depth] = 1.0

        in_touch = 1 - not_in_touch

        return not_in_touch, in_touch

    def internal_shadow(self, elastomer_depth):
        elastomer_depth_inv = self.max_depth - elastomer_depth
        elastomer_depth_inv = np.interp(elastomer_depth_inv, (0, self.elastomer_thickness), (0.0, 1.0))
        return elastomer_depth_inv

    def gauss_texture(self, shape):
        row, col = shape
        mean = 0
        gauss = np.random.normal(mean, self.texture_sigma, (row, col))
        gauss = gauss.reshape(row, col)
        return np.stack([gauss, gauss, gauss], axis=2)

    def elastic_deformation(self, protrusion_depth):
        fat_gauss_size = 95
        thin_gauss_size = 95
        thin_gauss_pad = (fat_gauss_size - thin_gauss_size) // 2
        # - gkern2(gauss2_size, 12)
        fat_gauss_kernel = gkern2(fat_gauss_size, 9)
        thin_gauss_kernel = np.pad(gkern2(thin_gauss_size, 9), thin_gauss_pad)
        dog_kernel = fat_gauss_kernel - thin_gauss_kernel
        # show_panel([fat_gauss_kernel, thin_gauss_kernel])
        return cv2.filter2D(protrusion_depth, -1, - fat_gauss_kernel)

        # kernel = gkern2(self.kernel_size, self.sigma)
        # deformation = protrusion_depth
        #
        # deformation2 = protrusion_depth
        # kernel2 = gkern2(52, 9)
        #
        # for i in range(self.t):
        #     deformation_ = cv2.filter2D(deformation, -1, kernel)
        #     r = np.max(protrusion_depth) / np.max(deformation_) if np.max(deformation_) > 0 else 1
        #     deformation = np.maximum(r * deformation_, protrusion_depth)
        #
        #     deformation2_ = cv2.filter2D(deformation2, -1, kernel2)
        #     r = np.max(protrusion_depth) / np.max(deformation2_) if np.max(deformation2_) > 0 else 1
        #     deformation2 = np.maximum(r * deformation2_, protrusion_depth)
        #
        # for i in range(self.t):
        #     deformation_ = cv2.filter2D(deformation2, -1, kernel)
        #     r = np.max(protrusion_depth) / np.max(deformation_) if np.max(deformation_) > 0 else 1
        #     deformation2 = np.maximum(r * deformation_, protrusion_depth)
        #
        #
        # deformation_x = 2 * deformation  # - deformation2
        #
        # return deformation_x / 2
        # # return np.stack([deformation_x, deformation_x, deformation_x], axis=2) / 3

    def _spec_diff(self, lm_data, v, n):
        imd = lm_data['id']
        ims = lm_data['is']
        alpha = lm_data['alpha']
        lm = - lm_data['field']
        color = lm_data['color_map']

        # Shared calculations
        lm_n = dot(lm, n)
        Rm = 2.0 * lm_n[:, :, np.newaxis] * n - lm

        # diffuse component
        diffuse_l = lm_n * imd

        # specular component
        spec_l = (dot(Rm, v) ** alpha) * ims

        return (diffuse_l + spec_l)[:, :, np.newaxis] * color

    def generate(self, depth):
        # depth_raw = depth

        # elastic deformation
        if self.apply_elastic_deformation:
            protrusion_map = self.bkg_depth - depth
            elastic_deformation = self.elastic_deformation(protrusion_map)
            depth = np.minimum(depth, self.bkg_depth + elastic_deformation)
            # depth = self.bkg_depth + elastic_deformation

        # Depth-map to surface point-cloud
        cam_matrix = get_camera_matrix(depth.shape[::-1])

        # s_raw = np.array(get_cloud_from_depth(cam_matrix, depth_raw).points) \
        #     .reshape((depth.shape[0], depth.shape[1], 3))

        s = np.array(get_cloud_from_depth(cam_matrix, depth).points) \
            .reshape((depth.shape[0], depth.shape[1], 3))

        # plot_depth_lines(
        #     [s_raw, s],
        #     # -elastic_deformation[:,:,0] / 255.0,
        #     protrusion_map,
        #     s.shape[0] // 2 + 60,
        #     legends=['Base depth', 'DoG depth']
        # )

        # Optical Rays (s - 0),
        # point from (0, 0, 0), the camera origin,
        # in the direction of pixel's point
        optical_rays = normalize_vectors(s)

        # Apply elastic deformation to the membrane
        # if self.apply_elastic_deformation:
        #     s_delta = self.s_ref - s
        #     # s_sharp = s
        #     protrusion_map = np.linalg.norm(s_delta, axis=2)
        #     elastic_deformation = self.elastic_deformation(protrusion_map)
        #     s = self.s_ref - elastic_deformation * optical_rays

        # show_field(s, n)

        # Add Random Gauss texture to the elastomer surface
        # gauss_texture = self.gauss_texture(s.shape[0:2])
        # s += gauss_texture * optical_rays

        # Phong's illumination vectors (n, v) calculations
        dx = normalize_vectors(derivative(s, 'x'))
        dy = normalize_vectors(derivative(s, 'y'))
        n = - np.cross(dx, dy)
        v = - optical_rays

        # print('xx')
        # show_field(s, n, 'red', subsample=99)

        I = self.background_img * self.ia \
            + np.sum([self._spec_diff(lm, v, n) for lm in self.lights], axis=0)

        I_rgb = (I * 255.0)
        I_rgb[I_rgb > 255.0] = 255.0
        I_rgb[I_rgb < 0.0] = 0.0
        I_rgb = I_rgb.astype(np.uint8)

        return I_rgb
