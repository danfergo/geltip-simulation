#!/usr/bin/env python
import cv2
import numpy as np

from sim_model.utils.camera import get_camera_matrix, depth2cloud
from sim_model.utils.maths import normalize_vectors, gkern2, dot_vectors, partial_derivative, normals, proj_vectors, \
    norm_vectors
from sim_model.utils.vis_img import to_normed_rgb
from sim_model.utils.vis_mesh import show_field

""" 
    GelSight Simulation
"""


class SimulationModel:

    def __init__(self, **config):
        self.default_ks = 0.15
        self.default_kd = 0.5
        self.default_alpha = 100
        self.ia = config['ia'] or 0.8
        self.fov = config['fov'] or 90

        self.lights = config['light_sources']

        self.bkg_depth = config['background_depth']
        self.cam_matrix = get_camera_matrix(self.bkg_depth.shape[::-1], self.fov)

        self.background_img = config['background_img']
        self.s_ref = depth2cloud(self.cam_matrix, self.bkg_depth)  # config['cloud_map']
        self.s_ref_n = normals(self.s_ref)

        self.apply_elastic_deformation = config['elastic_deformation'] if 'elastic_deformation' in config else False
        # self.elastomer_thickness = config['elastomer_thickness']
        # self.min_depth = config['min_depth']

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
        # self.max_depth = self.min_depth + self.elastomer_thickness

    @staticmethod
    def load_assets(assets_path, input_res, output_res, lf_method, n_light_sources):
        prefix = str(input_res[1]) + 'x' + str(input_res[0])

        # cloud = np.load(assets_path + '/' + prefix + '_ref_cloud.npy')
        # cloud = cloud.reshape((input_res[1], input_res[0], 3))
        # cloud = cv2.resize(cloud, output_res)

        # normals = np.load(assets_path + '/' + prefix + '_surface_normals.npy')
        # normals = normals.reshape((input_res[1], input_res[0], 3))
        # normals = cv2.resize(normals, output_res)
        light_fields = [
            # normalize_vectors(
            cv2.resize(
                # cv2.GaussianBlur(
                # cv2.resize(
                np.load(assets_path + '/' + lf_method + '_' + prefix + '_field_' + str(l) + '.npy'),
                # (80, 60), interpolation=cv2.INTER_LINEAR),
                # (25, 25), 0),
                output_res, interpolation=cv2.INTER_LINEAR)
            # )
            for l in range(n_light_sources)
        ]
        # normals,
        return light_fields

    # def protrusion_map(self, original, not_in_touch):
    #     protrusion_map = np.copy(original)
    #     protrusion_map[not_in_touch >= self.max_depth] = self.max_depth
    #     return protrusion_map

    # def segments(self, depth_map):
    #     not_in_touch = np.copy(depth_map)
    #     not_in_touch[not_in_touch < self.max_depth] = 0.0
    #     not_in_touch[not_in_touch >= self.max_depth] = 1.0
    #
    #     in_touch = 1 - not_in_touch
    #
    #     return not_in_touch, in_touch

    # def internal_shadow(self, elastomer_depth):
    #     elastomer_depth_inv = self.max_depth - elastomer_depth
    #     elastomer_depth_inv = np.interp(elastomer_depth_inv, (0, self.elastomer_thickness), (0.0, 1.0))
    #     return elastomer_depth_inv

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
        fat_gauss_kernel = gkern2(55, 5)
        # thin_gauss_kernel = np.pad(gkern2(thin_gauss_size, 21), thin_gauss_pad)
        # dog_kernel = fat_gauss_kernel - thin_gauss_kernel
        # show_panel([fat_gauss_kernel, thin_gauss_kernel])

        return cv2.filter2D(protrusion_depth, -1, fat_gauss_kernel)

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

    def _spec_diff(self, lm_data, v, n, s):
        imd = lm_data['id']
        ims = lm_data['is']
        alpha = lm_data['alpha']

        lm = - lm_data['field']  # points in the direction of the light source,
        color = lm_data['color_map']
        # lm = lm - proj_vectors(lm, self.s_ref_n) # - (self.s_ref - s)
        # print('-->', dot_vectors(lm, self.s_ref_n)[100, 100])
        # i.e. p(s) -> light source

        # show_field(cloud_map=s, field=lm, field_color='red', subsample=99)

        # Shared calculations
        lm_n = dot_vectors(lm, n)
        lm_n[lm_n < 0.0] = 0.0
        Rm = 2.0 * lm_n[:, :, np.newaxis] * n - lm

        # diffuse component
        diffuse_l = lm_n * imd

        # specular component
        spec_l = (dot_vectors(Rm, v) ** alpha) * ims

        return (diffuse_l + spec_l)[:, :, np.newaxis] * color

    def generate(self, depth):
        s = depth2cloud(self.cam_matrix, depth)

        # elastic deformation
        # if self.apply_elastic_deformation:
        #     protrusion_map = self.bkg_depth - depth
        #
        #     protrusion_map_bin = protrusion_map.copy()
        #     protrusion_map_bin[protrusion_map_bin > 1e-6] = 1
        #     protrusion_map_bin = cv2.dilate(protrusion_map_bin, np.ones((10, 10), np.uint8), iterations=1)
        #
        #     elastic_deformation = self.elastic_deformation(protrusion_map + 0.0001 * protrusion_map_bin)
        #
        #     inv_protrusion_map_bin = 1 - protrusion_map_bin
        #
        #     inv_protrusion_map_bin = cv2.filter2D(inv_protrusion_map_bin, -1, gkern2(55, 5))
        #
        #     protrusion_map_bin = cv2.dilate(protrusion_map_bin, np.ones((5, 5), np.uint8), iterations=1)
        #
        #     # elastic_depth = np.minimum(depth, depth - elastic_deformation)
        #     # elastic_depth = np.minimum(depth, self.bkg_depth + elastic_deformation)
        #
        #     dx = partial_derivative(protrusion_map_bin, 'x')
        #     dy = partial_derivative(protrusion_map_bin, 'y')
        #     ds = abs(dx) + abs(dy)
        #     ds = cv2.filter2D(ds, -1, gkern2(55, 5))

        # kernel =
        # dilation[dilation > 1e-3] = 1

        # fat_gauss_kernel = gkern2(55, 3)

        # cv2.imshow('xxx', to_normed_rgb(ds))
        # cv2.imshow('ds', to_normed_rgb(protrusion_map_bin))
        # cv2.imshow('ds', to_normed_rgb(elastic_def))
        # cv2.imshow('xxx', to_normed_rgb(protrusion_map))
        # cv2.imshow('asd', to_normed_rgb(elastic_deformation))
        # depth = elastic_depth
        # + 0.1 * elastic_def - 0.35 * elastic_deformation
        # elastic_depth = np.minimum(depth, self.bkg_depth - elastic_deformation)
        # elastic_depth = self.bkg_depth - elastic_deformation
        # elastic_s = depth2cloud(self.cam_matrix, elastic_depth)

        # print('asdasd', (elastic_def.max()))

        # s_elastic = depth2cloud(self.cam_matrix, depth)
        # s_delta = s_elastic - s
        # s_ = s
        # s = elastic_s

        # Optical Rays = s - 0
        optical_rays = normalize_vectors(s)

        # Apply elastic deformation to the membrane, over clouds
        # if self.apply_elastic_deformation:
        #     s_delta = self.s_ref - s
        #     # s_sharp = s
        #     protrusion_map = np.linalg.norm(s_delta, axis=2)
        #     elastic_deformation = self.elastic_deformation(protrusion_map)
        #     s = self.s_ref - elastic_deformation * optical_rays

        # Add Random Gauss texture to the elastomer surface
        # gauss_texture = self.gauss_texture(s.shape[0:2])
        # s += gauss_texture * optical_rays

        # Phong's illumination vectors (n, v) calculations
        n = - normals(s)

        v = - optical_rays

        I = self.background_img * self.ia \
            + np.sum([self._spec_diff(lm, v, n, s) for lm in self.lights], axis=0)

        I_rgb = (I * 255.0)
        I_rgb[I_rgb > 255.0] = 255.0
        I_rgb[I_rgb < 0.0] = 0.0
        I_rgb = I_rgb.astype(np.uint8)

        # plot_depth_lines(
        #     [elastic_s, s_],
        #     depth,
        #     row=s.shape[0] // 2 + 10,
        #     rgb_frame=I_rgb,
        #     legends=['Base depth', 'DoG depth']
        # )

        return I_rgb
