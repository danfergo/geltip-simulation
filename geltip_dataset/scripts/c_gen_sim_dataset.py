import os
import numpy as np
import cv2

from sim_model import SimulationModel
from sim_model import circle_mask


def main():
    fields_size = (120, 160)
    sim_size = (480, 640)

    n_fields = 3
    field_names = [
        'linear',
        'plane',
        # 'geodesic',
        'transport',
        # 'rlinear',
        # 'rplane',
        # 'rgeodesic',
        # 'rtransport'
    ]  # all fields

    objects = ['cone', 'sphere', 'random', 'cylinder', 'cylinder_shell', 'pacman', 'dot_in', 'dots']
    # objects = ['pacman']
    fields = ['transport']  # that actually are going to be generated
    # fields = field_names  # compute all.

    N_ROWS = 3
    N_CONTACTS = 6

    mask = circle_mask(sim_size[::-1])
    mask3 = np.stack([mask, mask, mask], axis=2)
    bkg_solid = np.ones(sim_size + (3,), dtype=np.float32) * np.array([[[0.6, 0.50, 0.96]]])

    dataset_path = os.path.dirname(os.path.abspath(__file__)) + '/../dataset/'
    assets_path = os.path.dirname(os.path.abspath(__file__)) + '/../../experimental_setup/geltip/sim_model/assets/'

    assets = {n: SimulationModel.load_assets(assets_path, fields_size, sim_size[::-1], n, n_fields) for n in
              field_names}

    light_coeffs = [
        {'color': [87, 159, 233], 'id': 0.8, 'is': 0.05},  # blue
        {'color': [197, 226, 241], 'id': 0.4, 'is': 0.05},  # green
        {'color': [196, 94, 255], 'id': 0.8, 'is': 0.05},  # red
        # {'color': [104, 175, 255], 'id': 0.8, 'is': 0.1},  # blue  # [120, 255, 153]
        # {'color': [154, 144, 255], 'id': 0.5, 'is': 0.1},  # green # [255, 130, 115]
        # {'color': [196, 94, 255], 'id': 0.8, 'is': 0.1},  # red # [108, 82, 255]
    ]

    light_sources = {
        n: [
            {'field': assets[n][1][l], **light_coeffs[l]} for l in range(len(light_coeffs))
        ] for n in field_names
    }
    # light_sources['combined'] = light_sources['linear'] + light_sources['geodesic']
    cloud = assets['linear'][0]

    for field in fields:
        for elastic_deformation in [True]:

            for use_bkg_rgb in [True, False]:

                set_name = 'sim_' \
                           + field \
                           + ('_elastic' if elastic_deformation else '') \
                           + ('_bkg' if use_bkg_rgb else '')

                if not os.path.exists(dataset_path + set_name):
                    os.mkdir(dataset_path + set_name)
                    [os.mkdir(dataset_path + set_name + '/' + obj) for obj in objects]

                for obj in objects:

                    if use_bkg_rgb:
                        bkg_bgr = cv2.imread(dataset_path + 'real_rgb_aligned/' + obj + '/bkg.png')
                        bkg_rgb = cv2.cvtColor(bkg_bgr, cv2.COLOR_BGR2RGB)
                        bkg_rgb = bkg_rgb * mask3 / 225.0
                        bkg = bkg_rgb
                    else:
                        bkg = bkg_solid

                    print('generating image using the field: ', field, ' for ', obj)

                    model = SimulationModel(**{
                        'ia': 0.8,
                        'fov': 90,
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

                            filename = dataset_path + set_name + '/' + obj + '/' + str(i) + '_' + str(j) + '.png'
                            cv2.imwrite(filename, cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB) * mask3)
                            print('saving: ' + filename)
                        # except Exception:
                        #     print('FAILED. ' + obj + '/' + str(i) + '_' + str(j) + '.npy')


if __name__ == '__main__':
    main()
