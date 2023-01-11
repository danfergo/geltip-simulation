import os
import numpy as np
import cv2

from experimental_setup.geltip.sim_model.model import SimulationModel
from experimental_setup.geltip.sim_model.scripts.utils.camera import circle_mask


def main():
    from experimental_setup.geltip.sim_model.scripts.utils.vis import show_panel
    fields_size = (160, 120)
    sim_size = (640, 480)

    mask = circle_mask(sim_size)
    mask3 = np.stack([mask, mask, mask], axis=2)
    bkg_zeros = np.zeros(sim_size[::-1] + (3,), dtype=np.float32)

    dataset_path = os.path.dirname(os.path.abspath(__file__)) + '/../dataset/'
    assets_path = os.path.dirname(os.path.abspath(__file__)) + '/../../experimental_setup/geltip/sim_model/assets/'

    cloud, linear_light_fields = SimulationModel.load_assets(assets_path, fields_size, sim_size, 'linear', 3)
    _, geodesic_light_fields = SimulationModel.load_assets(assets_path, fields_size, sim_size, 'geodesic', 3)
    _, rgeodesic_light_fields = SimulationModel.load_assets(assets_path, fields_size, sim_size, 'rgeodesic', 3)

    light_coeffs = [
        {'color': [196, 94, 255], 'id': 0.5, 'is': 0.1},  # red # [108, 82, 255]
        {'color': [154, 144, 255], 'id': 0.5, 'is': 0.1},  # green # [255, 130, 115]
        {'color': [104, 175, 255], 'id': 0.5, 'is': 0.1},  # blue  # [120, 255, 153]
    ]

    light_sources = {
        'linear': [{'field': linear_light_fields[l], **light_coeffs[l]} for l in range(3)],
        'geodesic': [{'field': geodesic_light_fields[l], **light_coeffs[l]} for l in range(3)],
        'rgeodesic': [{'field': rgeodesic_light_fields[l], **light_coeffs[l]} for l in range(3)],
    }
    light_sources['combined'] = light_sources['linear'] + light_sources['geodesic']

    objects = ['cone', 'sphere', 'random', 'cylinder', 'cylinder_shell', 'pacman', 'dot_in', 'dots']
    # fields = ['linear', 'geodesic', 'combined']
    fields = ['rgeodesic']

    N_ROWS = 3
    N_CONTACTS = 6

    for field in fields:
        for obj in objects:
            bkg_rgb = (cv2.cvtColor(
                cv2.imread(dataset_path + 'real_rgb_aligned/' + obj + '/bkg.png'),
                cv2.COLOR_BGR2RGB)) * mask3 / 225.0
            for elastic_deformation in [False, True]:
                for use_bkg_rgb in [True, False]:
                    bkg = bkg_rgb if use_bkg_rgb else bkg_zeros

                    print('generating image using the field: ', field, ' for ', obj)

                    model = SimulationModel(**{
                        'ia': 0.8,
                        'light_sources': light_sources[field],
                        'background_depth': np.load(assets_path + 'bkg.npy'),
                        'cloud_map': cloud,
                        'background_img': bkg,
                        'elastomer_thickness': 0.004,
                        'min_depth': 0.026,
                        'texture_sigma': 0.000005,
                        'elastic_deformation': elastic_deformation
                    })

                    for i in range(N_ROWS):
                        for j in range(N_CONTACTS):
                            # try:
                            depth_map = np.load(
                                dataset_path + 'sim_depth/' + obj + '/' + str(i) + '_' + str(j) + '.npy')
                            rgb_real = (cv2.cvtColor(
                                cv2.imread(
                                    dataset_path + 'real_rgb_aligned/' + obj + '/' + str(i) + '_' + str(j) + '.png'),
                                cv2.COLOR_BGR2RGB)).astype(np.uint8)

                            rgb_real = (rgb_real * mask3).astype(np.uint8)
                            rgb = model.generate(depth_map)

                            # show_panel(
                            #     [
                            #         # to_normed_rgb(depth_map),
                            #         rgb_real,
                            #         rgb
                            #     ],
                            #     (1, 2))

                            filename = dataset_path + 'sim_' \
                                       + field \
                                       + ('_elastic' if elastic_deformation else '') \
                                       + ('_bkg' if use_bkg_rgb else '') \
                                       + '/' + obj + '/' + str(i) + '_' + str(
                                j) + '.png'
                            cv2.imwrite(filename, cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB) * mask3)
                            print('saving: ' + filename)
                        # except Exception:
                        #     print('FAILED. ' + obj + '/' + str(i) + '_' + str(j) + '.npy')


if __name__ == '__main__':
    main()
