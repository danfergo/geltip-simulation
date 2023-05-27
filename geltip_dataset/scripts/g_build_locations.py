from os import path
import numpy as np

import cv2
import yaml

from sim_model.utils.camera import circle_mask


def normalize_depth_map(d):
    d = d - np.min(d)
    return d / np.max(d)


def depthimg2Meters(depth):
    extent = 2.0
    znear = 0.0020000000949949026
    zfar = 100.0

    near = znear * extent
    far = zfar * extent
    image = near / (1 - depth * (1 - near / far))
    return image


def main():
    dataset_path = '/home/danfergo/Projects/PhD/geltip_simulation/geltip_dataset/dataset/'
    objects = ['cone', 'sphere', 'random', 'cylinder', 'cylinder_shell', 'pacman', 'dot_in', 'dots']

    sim_set = 'adepth'

    set_params = {
        # 18
        'depth': {
            'rows': 3,
            'contacts': 6,
            'align_real': True
        },
        # 3456
        'adepth': {
            'rows': 5,
            'contacts': 576,  # 36 printers.
            'align_real': False
        }
    }

    locations = {}

    c_mask = circle_mask((640, 480))

    for obj in objects:
        bkg_depth_path = dataset_path + f"sim_{sim_set}/{obj}/bkg.npy"
        bkg_depth = np.load(bkg_depth_path)

        print('object:' + obj)
        for i in range(set_params[sim_set]['rows']):
            print('row:' + str(i))
            for j in range(set_params[sim_set]['contacts']):
                depth_path = dataset_path + f"sim_{sim_set}/{obj}/{i}_{j}.npy"
                depth = np.load(depth_path)

                diff = (bkg_depth - depth) * c_mask
                seg_mask = diff
                seg_mask[seg_mask > 1e-5] = 1.0
                seg_mask[seg_mask < 1e-9] = 0.0

                where = np.where(seg_mask > 0.5)
                mean_point = round(np.mean(where[0])), round(np.mean(where[1]))

                seg_mask3 = (np.stack([seg_mask, seg_mask, seg_mask], axis=2) * 255).astype(np.uint8)

                # frame = seg_mask3
                # frame = cv2.circle(frame,
                #                    tuple(reversed(mean_point)),
                #                    10,
                #                    (0, 0, 255),
                #                    2)
                locations[f'{obj}/{i}_{j}'] = list(mean_point)

                # cv2.imshow('frame', frame)
                # cv2.waitKey(-1)
    yaml.dump(locations, open(path.join(dataset_path, f'{sim_set}_object_locations.yaml'), 'w'))


if __name__ == '__main__':
    main()
