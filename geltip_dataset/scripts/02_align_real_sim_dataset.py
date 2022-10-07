import os
import cv2
import numpy as np

from experimental_setup.geltip.sim_model.scripts.utils.camera import circle_mask
from experimental_setup.geltip.sim_model.scripts.utils.vis import show_panel, to_normed_rgb


def in_contact_mask(bkg_depth, depth):
    diff = bkg_depth - depth
    diff[diff < 1e-05] = 0
    # diff[diff > 0] = 1
    return to_normed_rgb(diff)


def main():
    dataset_path = os.path.dirname(os.path.abspath(__file__)) + '/../dataset/'
    assets_path = os.path.dirname(os.path.abspath(__file__)) + '/../../experimental_setup/geltip/sim_model/assets/'

    objects = ['cone', 'sphere', 'random', 'cylinder', 'cylinder_shell', 'pacman', 'dot_in', 'dots']

    show_alignment_panel = True
    N_ROWS = 3
    N_CONTACTS = 6

    mask = circle_mask((640, 480))
    mask3 = np.stack([mask, mask, mask], axis=2)

    def align(key):
        bkg_depth = np.load(dataset_path + 'sim_depth/cylinder/bkg.npy')

        rgb = cv2.cvtColor(cv2.imread(dataset_path + 'real_rgb/' + obj + '/' + key + '.png'),
                           cv2.COLOR_BGR2RGB)
        depth = np.load(dataset_path + 'sim_depth/' + obj + '/' + key + '.npy')

        in_contact_rgb = in_contact_mask(bkg_depth, depth)

        depth = to_normed_rgb(depth)

        # manual alignment
        height, width = rgb.shape[:2]
        center = (width / 2, height / 2)

        align_matrix = cv2.getRotationMatrix2D(center=center, angle=195, scale=1.15)
        rgb_aligned = cv2.warpAffine(src=rgb, M=align_matrix, dsize=(width, height))

        if show_alignment_panel:
            diff_aligned = (rgb_aligned * 0.5 + in_contact_rgb * 0.5).astype(np.uint8)

            show_panel([rgb, in_contact_rgb, rgb_aligned, diff_aligned], (2, 2))

        rgb_aligned = cv2.cvtColor(rgb_aligned, cv2.COLOR_BGR2RGB)
        # cv2.imwrite(dataset_path + 'real_rgb_aligned/' + obj + '/' + key + '.png', rgb_aligned * mask3)

    # show
    for obj in objects:

        align('bkg')

        for i in range(N_ROWS):
            for j in range(N_CONTACTS):
                align(str(i) + '_' + str(j))
        print('ended ' + obj)

    print('ended all.')


if __name__ == '__main__':
    main()
