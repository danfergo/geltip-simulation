from os import path
import numpy as np

import cv2
import yaml


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
    N_ROWS = 3
    N_CONTACTS = 6
    locations = {}
    for obj in objects:
        bkg_depth_path = dataset_path + f"sim_depth/cylinder/bkg.npy"
        bkg_depth = np.load(bkg_depth_path)

        for i in range(N_ROWS):
            for j in range(N_CONTACTS):
                depth_path = dataset_path + f"sim_depth/{obj}/{i}_{j}.npy"
                depth = np.load(depth_path)

                diff = bkg_depth - depth
                seg_mask = diff
                seg_mask[seg_mask > 1e-7] = 1.0
                seg_mask[seg_mask < 1e-7] = 0.0

                where = np.where(seg_mask > 0.5)
                mean_point = round(np.mean(where[0])), round(np.mean(where[1]))

                seg_mask3 = (np.stack([seg_mask, seg_mask, seg_mask], axis=2) * 255).astype(np.uint8)

                frame = seg_mask3
                frame = cv2.circle(frame,
                                   tuple(reversed(mean_point)),
                                   10,
                                   (0, 0, 255),
                                   2)
                locations[f'{obj}/{i}_{j}'] = list(mean_point)

                cv2.imshow('frame', frame)
                # cv2.waitKey(-1)
    yaml.dump(locations, open(path.join(dataset_path, 'object_locations.yaml'), 'w'))


if __name__ == '__main__':
    main()
