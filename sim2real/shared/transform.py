import imgaug.augmenters as iaa

from dfgiatk.ops.img import cvt_batch, CVT_HWC2CHW

import numpy as np


# """ sim model stuff """
# dataset_path = e.data_path
# fields_size = (160, 120)
# sim_size = (640, 480)
#
# mask = circle_mask(sim_size)
# mask3 = np.stack([mask, mask, mask], axis=2)
#
# light_coeffs = [
#     {'color': [196, 94, 255], 'id': 0.5, 'is': 0.1},  # red # [108, 82, 255]
#     {'color': [154, 144, 255], 'id': 0.5, 'is': 0.1},  # green # [255, 130, 115]
#     {'color': [104, 175, 255], 'id': 0.5, 'is': 0.1},  # blue  # [120, 255, 153]
# ]
#
# cloud, linear_light_fields = SimulationModel.load_assets(e.assets_path, fields_size, sim_size, 'linear', 3)
# _, geodesic_light_fields = SimulationModel.load_assets(e.assets_path, fields_size, sim_size, 'geodesic', 3)
#
# light_sources = {
#     'linear': [{'field': linear_light_fields[l], **light_coeffs[l]} for l in range(3)],
#     'geodesic': [{'field': geodesic_light_fields[l], **light_coeffs[l]} for l in range(3)],
# }
# light_sources['combined'] = light_sources['linear'] + light_sources['geodesic']
#
# bkgs = {
#     cls: (cv2.cvtColor(
#         cv2.imread(dataset_path + 'real_rgb_aligned/' + cls + '/bkg.png'),
#         cv2.COLOR_BGR2RGB)) * mask3 / 225.0
#     for cls in ['cone', 'sphere', 'random', 'cylinder', 'cylinder_shell', 'pacman', 'dot_in', 'dots']
# }
#
# bkg_depth = np.load(e.assets_path + 'bkg.npy')
#
#
# # bkg = bkg_rgb if use_bkg_rgb else bkg_zeros
#
# def sim_model(bkg):
#     elastic_deformation = bool(getrandbits(1))
#     f = randint(0, 2)
#     field = 'geodesic' if f == 0 else ('linear' if f == 2 else 'combined')
#
#     return SimulationModel(**{
#         'ia': 0.8,
#         'light_sources': light_sources[field],
#         'background_depth': bkg_depth,
#         'cloud_map': cloud,
#         'background_img': bkg,
#         'elastomer_thickness': 0.004,
#         'min_depth': 0.026,
#         'texture_sigma': 0.000005,
#         'elastic_deformation': elastic_deformation
#     })
#
#
# """ end sim model stuff """
#
# if regen_tactile:
#     obj_classes = [path.basename(path.dirname(s)) for s in samples]
#     print('-->', xs[0, :].shape)
#
#     # gen
#     #
#     xs = np.array(
#         [
#             sim_model(bkgs[obj_classes[i]]).generate(xs[i, :] + 0.001 * rnd_bkgs[randint(0, n_rnd_bkgs - 1)])
#             for i in range(len(xs))
#         ]
#     )
#
#     # align
#     xs = np.array(
#         [cv2.warpAffine(src=xs[i], M=align_matrix, dsize=(width, height)) * mask3 for i in range(len(xs))]
#     )
#     for i in range(len(xs)):
#         cv2.imshow('frame', xs[i].astype(np.uint8))
#         cv2.waitKey(-1)

# elastic_deformation = True
# use_bkg_rgb = True
# field = 'geodesic'
#
# height, width = 480, 640
# center = (width / 2, height / 2)
#
# align_matrix = cv2.getRotationMatrix2D(center=center, angle=195, scale=1.15)
#
# rnd_bkgs = []
# n_rnd_bkgs = 12
# for i in range(n_rnd_bkgs):
#     rnd_bkgs.append(
#         (cv2.resize(
#             cv2.imread(e.data_path + '/../textures/' + str(i + 1) + '.png', cv2.IMREAD_GRAYSCALE),
#             dsize=(640, 480)
#         ) / 225.0).astype(np.float32)
#     )
#

def transform(image_aug=False, regen_tactile=False):
    seq = iaa.Sequential(
        [
            iaa.Resize({"height": 120, "width": "keep-aspect-ratio"}),
        ] + ([
                 iaa.OneOf([
                     # iaa.Affine(rotate=45),
                     iaa.AdditiveGaussianNoise(scale=0.7),
                     iaa.Add(50, per_channel=True),
                     iaa.Sharpen(alpha=0.5)
                 ])
             ] if image_aug else [])
    )

    def _(xs, samples):
        torch_images = xs.astype(np.float32) / 255.0
        torch_images = seq(images=torch_images)

        torch_images = cvt_batch(torch_images, CVT_HWC2CHW).astype(np.float32)
        return torch_images

    return _
