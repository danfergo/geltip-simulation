#!/usr/bin/env python
import cv2
import numpy as np

import open3d as o3d

from sim_model.utils.camera import get_camera_matrix, depth2cloud
from sim_model.utils.maths import normalize_vectors, gkern2, dot_vectors, normals, proj_vectors, partial_derivative
from sim_model.utils.vis_img import to_normed_rgb


class SimulationModel:

    def __init__(self, **config):
        self.default_ks = 0.15
        self.default_kd = 0.5
        self.default_alpha = 100
        self.ia = config['ia'] or 0.8
        self.fov = config['fov'] or 90

        self.lights = config['light_sources']
        self.rectify_fields = config['rectify_fields']

        self.bkg_depth = config['background_depth']
        self.cam_matrix = get_camera_matrix(self.bkg_depth.shape[::-1], self.fov)

        self.background_img = config['background_img']
        self.s_ref = depth2cloud(self.cam_matrix, self.bkg_depth)  # config['cloud_map']
        self.s_ref_n = normals(self.s_ref)

        self.apply_elastic_deformation = config['elastic_deformation'] if 'elastic_deformation' in config else False
        self.internal_shadow = config['internal_shadow'] if 'internal_shadow' in config else 0.15

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

    @staticmethod
    def load_assets(assets_path, input_res, output_res, lf_method, n_light_sources):
        prefix = str(input_res[1]) + 'x' + str(input_res[0])

        light_fields = [
            cv2.resize(
                cv2.GaussianBlur(
                    np.load(assets_path + '/' + lf_method + '_' + prefix + '_field_' + str(l) + '.npy'),
                    (25, 25), 0),
                output_res, interpolation=cv2.INTER_LINEAR)
            # )
            for l in range(n_light_sources)
        ]
        # normals,
        return light_fields

    def gauss_texture(self, shape):
        row, col = shape
        mean = 0
        gauss = np.random.normal(mean, self.texture_sigma, (row, col))
        gauss = gauss.reshape(row, col)
        return np.stack([gauss, gauss, gauss], axis=2)

    def _spec_diff(self, lm_data, v, n, s):
        imd = lm_data['id']
        ims = lm_data['is']
        alpha = lm_data['alpha']

        lm = - lm_data['field']
        color = lm_data['color_map']

        if self.rectify_fields:
            lm = normalize_vectors(lm - proj_vectors(lm, self.s_ref_n))

        # Shared calculations
        lm_n = dot_vectors(lm, n)
        lm_n[lm_n < 0.0] = 0.0
        Rm = 2.0 * lm_n[:, :, np.newaxis] * n - lm

        # diffuse component
        diffuse_l = lm_n * imd

        # specular component
        spec_l = (dot_vectors(Rm, v) ** alpha) * ims

        return (diffuse_l + spec_l)[:, :, np.newaxis] * color

    def calculate_occluded_areas(self, protrusion_map, optical_rays):
        # Threshold the protrusion_map to create a binary map
        binary_map = (protrusion_map > 0.00001).astype(np.float32)

        # Compute the partial derivatives of the binary map
        areas_x = partial_derivative(binary_map, 'x')
        areas_y = partial_derivative(binary_map, 'y')

        # Compare the signs of the optical rays and partial derivatives
        sign_comparison = np.equal(np.sign(optical_rays[:, :, :2]), np.sign(np.stack([areas_x, areas_y], axis=-1)))

        # Calculate the occluded areas
        occluded_areas = np.clip(sign_comparison.sum(axis=-1) / 0.05, 0, 1)

        # Dilate the occluded areas
        kernel = np.ones((3, 3), np.uint8)
        occluded_areas = cv2.dilate(occluded_areas, kernel, iterations=1)

        # Apply a Gaussian filter
        occluded_areas = cv2.filter2D(occluded_areas, -1, gkern2(55, 5))

        # Normalize the occluded areas
        occluded_areas = (occluded_areas - occluded_areas.min()) / (occluded_areas.max() - occluded_areas.min())

        # Remove regions where the binary map has a value of 1
        kernel = np.ones((7, 7), np.uint8)
        dilated_binary_map = cv2.dilate(binary_map, kernel, iterations=1)
        occluded_areas *= (1 - dilated_binary_map)

        return occluded_areas

    def calculate_occluded_areas_alternative(self, surface_normals, optical_rays, threshold=0.95):
        # Compute the dot product between surface normals and optical rays
        dot_product = np.abs(np.sum(surface_normals * optical_rays, axis=-1))

        # Threshold the dot product to create an occlusion map
        occlusion_map = (dot_product > threshold).astype(np.float32)

        # Dilate the occlusion map
        kernel = np.ones((3, 3), np.uint8)
        occlusion_map = cv2.dilate(occlusion_map, kernel, iterations=1)

        # Apply a Gaussian filter
        occlusion_map = cv2.GaussianBlur(occlusion_map, (55, 55), 5)

        # Normalize the occlusion map
        occlusion_map = (occlusion_map - occlusion_map.min()) / (occlusion_map.max() - occlusion_map.min())

        return occlusion_map

    def generate(self, depth):
        # Calculate the protrusion_map
        protrusion_map = self.bkg_depth - depth

        # surface point-cloud
        s = depth2cloud(self.cam_matrix, depth)

        # Optical Rays = s - 0
        optical_rays = normalize_vectors(s)

        # Calculate the occluded areas
        occluded_areas = self.calculate_occluded_areas(protrusion_map, optical_rays)
        print('areas', np.min(occluded_areas), np.max(occluded_areas))
        cv2.imshow('areas', to_normed_rgb(occluded_areas))
        cv2.waitKey(-1)

        # binary_map = np.where(protrusion_map > 0.000001, 1, 0).astype(np.float32)
        # areas_x = partial_derivative(binary_map, 'x')
        # areas_y = partial_derivative(binary_map, 'y')
        #
        # sign_or_x = np.sign(optical_rays[:, :, 0])
        # sign_or_y = np.sign(optical_rays[:, :, 1])
        # sign_a_x = np.sign(areas_x)
        # sign_a_y = np.sign(areas_y)
        #
        # occluded_areas = np.clip((np.equal(sign_a_x, sign_or_x).astype(np.float32) +
        #                  np.equal(sign_a_y, sign_or_y).astype(np.float32)) / 0.05, 0, 1)
        #
        # kernel = np.ones((3, 3), np.uint8)
        # occluded_areas = cv2.dilate(occluded_areas, kernel, iterations=1)
        # occluded_areas = cv2.filter2D(occluded_areas, -1, gkern2(55, 5))
        # occluded_areas = occluded_areas - np.min(occluded_areas)
        # occluded_areas = occluded_areas / np.max(occluded_areas)
        # occluded_areas = (1 - binary_map) * occluded_areas

        # elastic deformation (as in the paper, submitted to RSS)
        if self.apply_elastic_deformation:
            elastic_deformation = cv2.filter2D(self.bkg_depth - depth, -1, gkern2(55, 5))

            elastic_deformation = np.maximum((1 - occluded_areas) * elastic_deformation, np.zeros_like(occluded_areas))
            depth = np.minimum(depth, self.bkg_depth - elastic_deformation).astype(np.float32)
            print(depth.dtype)

        # surface point-cloud
        s = depth2cloud(self.cam_matrix, depth)

        # Optical Rays = s - 0
        optical_rays = normalize_vectors(s)

        # illumination vectors (n, v) calculations
        n = - normals(s)
        v = - optical_rays

        # Calculate the absolute difference between bkg_depth and depth using the precomputed protrusion_map
        contact_diff = np.abs(protrusion_map)

        # Clip the difference to a maximum value (e.g., 0.03)
        contact_diff = np.clip(contact_diff, 0, 0.03)

        # Normalize the difference to a percentage
        contact_percentage = contact_diff / 0.03

        # Multiply the internal_shadow by the percentage and clip it to a valid range (0 to 1)
        shadow_factor = np.clip(self.internal_shadow * contact_percentage, 0, 1)

        ambient_component = self.background_img * (self.ia - shadow_factor)[:, :, np.newaxis]

        I = ambient_component + np.sum([self._spec_diff(lm, v, n, s) for lm in self.lights], axis=0)

        I_rgb = (I * 255.0)
        I_rgb[I_rgb > 255.0] = 255.0
        I_rgb[I_rgb < 0.0] = 0.0
        I_rgb = I_rgb.astype(np.uint8)

        # Normalize the occluded_mask values to 0 to 255
        occluded_map = (occluded_areas.astype(np.float32) * 255).astype(np.uint8)

        # Convert occluded_map to 3-channel image
        occluded_map_3channel = cv2.cvtColor(occluded_map, cv2.COLOR_GRAY2BGR)

        # Overlay the occluded_map over I_rgb with 50% opacity
        I_rgb_overlay = cv2.addWeighted(I_rgb, 0.5, occluded_map_3channel, 0.5, 0)

        # Normalize the RGB values
        I_rgb_overlay = np.clip(I_rgb_overlay, 0, 255).astype(np.uint8)

        # Display the occluded_map and the overlay image
        # cv2.imshow('Occluded Map', occluded_map)
        cv2.imshow('Overlay Image', I_rgb_overlay)
        cv2.imshow('Image', I_rgb)
        cv2.waitKey(-1)

        return I_rgb
