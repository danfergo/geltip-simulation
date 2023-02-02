from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

import numpy as np
import os
import cv2

from sim_model import circle_mask


def rectified_mae_loss(true, test):
    assert true.shape[0] == test.shape[0], \
        "True and Test batches must have the same shape."
    assert true.dtype == test.dtype and (true.dtype == np.uint8 or true.dtype == np.float32), \
        "That's a wrong dtype" + str(true.dtype) + " " + str(test.dtype)

    pixel_range = 255 if test.dtype == np.uint8 else 1
    area = true.shape[1] * true.shape[2] * true.shape[3]

    losses = [np.sum(cv2.absdiff(true[i] / pixel_range, test[i] / pixel_range)) / area for i in range(true.shape[0])]

    return sum(losses) / len(losses)


# -1 to 1, 1 means the images are identical
def ssim_loss(true, test):
    assert true.shape[0] == test.shape[0], "True and Test batches must have the same shape."
    losses = [ssim(true[i], test[i], channel_axis=2) for i in range(true.shape[0])]
    return sum(losses) / true.shape[0]


def psnr_loss(true, test):
    assert true.shape[0] == test.shape[0], "True and Test batches must have the same shape."
    losses = [psnr(true, test) for i in range(true.shape[0])]
    return sum(losses) / true.shape[0]


def combined_loss(true, test):
    return rectified_mae_loss(true, test) \
           + (1 - ssim_loss(true, test)) \
           + psnr_loss(true, test)


def main():
    dataset_path = os.path.dirname(os.path.abspath(__file__)) + '/../dataset/'

    objects = ['cone', 'sphere', 'random', 'cylinder', 'cylinder_shell', 'pacman', 'dot_in', 'dots']
    field_names = ['linear', 'plane', 'geodesic', 'transport']
    # field_names = ['plane']
    fields = [
        f'{n}{"_elastic" if elastic else ""}{"_bkg" if bkg else ""}'
        for n in field_names for elastic in [True] for bkg in [True, False]
    ]

    N_ROWS = 3
    N_CONTACTS = 6

    # show
    mask = circle_mask((640, 480))
    mask3 = np.stack([mask, mask, mask], axis=2)

    for field in fields:
        real_imgs = []
        sim_imgs = []

        for obj in objects:
            for i in range(N_ROWS):
                for j in range(N_CONTACTS):
                    real_img = (cv2.cvtColor(
                        cv2.imread(dataset_path + 'real_rgb_aligned/' + obj + '/' + str(i) + '_' + str(j) + '.png'),
                        cv2.COLOR_BGR2RGB
                    ) * mask3).astype(np.uint8)

                    sim_img = cv2.cvtColor(
                        cv2.imread(dataset_path + 'sim_' + field + '/' + obj + '/' + str(i) + '_' + str(j) + '.png'),
                        cv2.COLOR_BGR2RGB
                    )

                    real_imgs.append(real_img)
                    sim_imgs.append(sim_img)

        real_imgs = np.array(real_imgs)
        sim_imgs = np.array(sim_imgs)

        mae_err = rectified_mae_loss(real_imgs, sim_imgs)
        ssim_err = ssim_loss(real_imgs, sim_imgs)
        psnr_err = psnr_loss(real_imgs, sim_imgs)

        print('---------')
        print(field + ' (rmae): ' + "{:.3f}".format(mae_err))
        print(field + ' (ssim): ' + "{:.3f}".format(ssim_err))
        print(field + ' (psnr): ' + "{:.3f}".format(psnr_err))

        # show_panel([real_img, sim], (1, 2))


if __name__ == '__main__':
    main()
