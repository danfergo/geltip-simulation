import os
import cv2
import numpy as np

from sim_model.utils.camera import circle_mask
from sim_model.utils.vis_img import to_normed_rgb, show_panel


def in_contact_mask(bkg_depth, depth):
    diff = bkg_depth - depth
    diff[diff < 1e-05] = 0
    # diff[diff > 0] = 1
    return to_normed_rgb(diff)


def main():
    dataset_path = os.path.dirname(os.path.abspath(__file__)) + '/../dataset/'
    assets_path = os.path.dirname(os.path.abspath(__file__)) + '/../../experimental_setup/geltip/sim_model/assets/'

    objects = [
        'cone',
        'sphere',
        'random',
        'cylinder',
        'cylinder_shell',
        'pacman',
        'dot_in',
        'dots'
    ]

    show_alignment_panel = False

    sim_set = 'depth'
    # sim_set = 'adepth'

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

    mask = circle_mask((640, 480))
    mask3 = np.stack([mask, mask, mask], axis=2)

    if not os.path.exists(f'{dataset_path}/sim_{sim_set}_aligned'):
        os.mkdir(f'{dataset_path}/sim_{sim_set}_aligned')
        [os.mkdir(f'{dataset_path}/sim_{sim_set}_aligned/{obj}') for obj in objects]

    def align(obj, key, align_real):
        bkg_depth = np.load(f'{dataset_path}/sim_{sim_set}/{obj}/bkg.npy')

        depth = np.load(f'{dataset_path}sim_{sim_set}/{obj}/{key}.npy')
        in_contact_rgb = in_contact_mask(bkg_depth, depth)

        if align_real:
            rgb = cv2.cvtColor(cv2.imread(f'{dataset_path}/real_rgb/{obj}/{key}.png'), cv2.COLOR_BGR2RGB)

            # manual alignment
            height, width = rgb.shape[:2]
            center = (width // 2, height // 2)

            rotation_matrix = cv2.getRotationMatrix2D(center=center, angle=192, scale=1.15)
            rgb_aligned = cv2.warpAffine(src=rgb, M=rotation_matrix, dsize=(width + 50, height + 50))

            tx = 0.0
            ty = -20.0
            translation_matrix = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty]])
            rgb_aligned = cv2.warpAffine(src=rgb_aligned, M=translation_matrix, dsize=(width, height))

            # M = cv2.matFromArray(2, 3, cv2.CV_64FC1, [1, 0, 50, 0, 1, 100]);
            #
            # depth_aligned = cv2.warpAffine(src=depth, M=rotation_matrix, dsize=(width, height))

            if show_alignment_panel:
                # in_contact_rgb = to_normed_rgb(in_contact_rgb)
                # print(rgb_aligned.shape, in_contact_rgb.shape)

                diff_aligned = (rgb_aligned * 0.5 + in_contact_rgb * 0.5).astype(np.uint8)
                diff_aligned = cv2.circle(diff_aligned, center, 10, (0, 255, 0), 2)
                show_panel([rgb, in_contact_rgb, rgb_aligned, diff_aligned], (2, 2))

            rgb_aligned = cv2.cvtColor(rgb_aligned, cv2.COLOR_BGR2RGB)
            # cv2.imwrite(dataset_path + 'real_rgb_aligned/' + obj + '/' + key + '.png', rgb_aligned * mask3)

        np.save(open(f'{dataset_path}/sim_{sim_set}_aligned/{obj}/{key}.npy', 'wb'), depth * mask)

    # show
    for obj in objects:
        align(obj, 'bkg', set_params[sim_set]['align_real'])

        for i in range(set_params[sim_set]['rows']):
            for j in range(set_params[sim_set]['contacts']):
                align(obj, str(i) + '_' + str(j), set_params[sim_set]['align_real'])
        print('ended ' + obj)

    print('ended all.')


if __name__ == '__main__':
    main()
