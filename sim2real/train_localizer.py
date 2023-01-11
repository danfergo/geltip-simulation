from random import randint, getrandbits, random

import cv2

from dfgiatk.experimenter import run, e, Logger, Validator, Plotter, EBoard, ModelSaver

from os import path
from torch import nn, optim
import numpy as np

from dfgiatk.ops.img import cvt_batch, CVT_HWC2CHW

from dfgiatk.loaders import DatasetSampler
from dfgiatk.loaders.image_loader import LocalizationLabeler, NumpyMapsLabeler, ImageLoader
from dfgiatk.models import resnet50

import imgaug.augmenters as iaa

from dfgiatk.train import fit_to_dataset
from experimental_setup.geltip.sim_model.model import SimulationModel
from experimental_setup.geltip.sim_model.scripts.utils.camera import circle_mask


def loader(partition, epoch_size, batch_size, set):
    samples = DatasetSampler.load_from_yaml(
        path.join(e.data_path, partition + '_split.yaml'),
        path.join(e.data_path, set)
    )
    loader = NumpyMapsLabeler(
        path.join(e.data_path, 'sim_depth'),  # _aligned
        transform=e[partition + '_transform']
    )

    # loader = ImageLoader(
    #     transform=e[partition + '_transform']
    # )

    labeler = LocalizationLabeler(
        locations_path=path.join(e.data_path, 'object_locations.yaml')
    )

    return DatasetSampler(
        samples,
        loader=loader,
        labeler=labeler,
        epoch_size=epoch_size,
        batch_size=batch_size
    )


def transform(image_aug=False, regen_tactile=False):
    """ sim model stuff """
    dataset_path = e.data_path
    fields_size = (160, 120)
    sim_size = (640, 480)

    mask = circle_mask(sim_size)
    mask3 = np.stack([mask, mask, mask], axis=2)

    light_coeffs = [
        {'color': [196, 94, 255], 'id': 0.5, 'is': 0.1},  # red # [108, 82, 255]
        {'color': [154, 144, 255], 'id': 0.5, 'is': 0.1},  # green # [255, 130, 115]
        {'color': [104, 175, 255], 'id': 0.5, 'is': 0.1},  # blue  # [120, 255, 153]
    ]

    cloud, linear_light_fields = SimulationModel.load_assets(e.assets_path, fields_size, sim_size, 'linear', 3)
    _, geodesic_light_fields = SimulationModel.load_assets(e.assets_path, fields_size, sim_size, 'geodesic', 3)

    light_sources = {
        'linear': [{'field': linear_light_fields[l], **light_coeffs[l]} for l in range(3)],
        'geodesic': [{'field': geodesic_light_fields[l], **light_coeffs[l]} for l in range(3)],
    }
    light_sources['combined'] = light_sources['linear'] + light_sources['geodesic']

    bkgs = {
        cls: (cv2.cvtColor(
            cv2.imread(dataset_path + 'real_rgb_aligned/' + cls + '/bkg.png'),
            cv2.COLOR_BGR2RGB)) * mask3 / 225.0
        for cls in ['cone', 'sphere', 'random', 'cylinder', 'cylinder_shell', 'pacman', 'dot_in', 'dots']
    }

    bkg_depth = np.load(e.assets_path + 'bkg.npy')

    # bkg = bkg_rgb if use_bkg_rgb else bkg_zeros

    def sim_model(bkg):
        elastic_deformation = bool(getrandbits(1))
        f = randint(0, 2)
        field = 'geodesic' if f == 0 else ('linear' if f == 2 else 'combined')

        return SimulationModel(**{
            'ia': 0.8,
            'light_sources': light_sources[field],
            'background_depth': bkg_depth,
            'cloud_map': cloud,
            'background_img': bkg,
            'elastomer_thickness': 0.004,
            'min_depth': 0.026,
            'texture_sigma': 0.000005,
            'elastic_deformation': elastic_deformation
        })

    """ end sim model stuff """

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
    elastic_deformation = True
    use_bkg_rgb = True
    field = 'geodesic'

    height, width = 480, 640
    center = (width / 2, height / 2)

    align_matrix = cv2.getRotationMatrix2D(center=center, angle=195, scale=1.15)

    rnd_bkgs = []
    n_rnd_bkgs = 12
    for i in range(n_rnd_bkgs):
        rnd_bkgs.append(
            (cv2.resize(
                cv2.imread(e.data_path + '/../textures/' + str(i + 1) + '.png', cv2.IMREAD_GRAYSCALE),
                dsize=(640, 480)
            ) / 225.0).astype(np.float32)
        )

    def _(xs, samples):
        if regen_tactile:
            obj_classes = [path.basename(path.dirname(s)) for s in samples]
            print('-->', xs[0, :].shape)

            # gen
            #
            xs = np.array(
                [
                    sim_model(bkgs[obj_classes[i]]).generate(xs[i, :]+ 0.001 * rnd_bkgs[randint(0, n_rnd_bkgs - 1)])
                    for i in range(len(xs))
                ]
            )

            # align
            xs = np.array(
                [cv2.warpAffine(src=xs[i], M=align_matrix, dsize=(width, height)) * mask3 for i in range(len(xs))]
            )
            for i in range(len(xs)):
                cv2.imshow('frame', xs[i].astype(np.uint8))
                cv2.waitKey(-1)

        torch_images = xs.astype(np.float32) / 255.0
        torch_images = seq(images=torch_images)

        torch_images = cvt_batch(torch_images, CVT_HWC2CHW).astype(np.float32)
        return torch_images

    return _


run(
    description="""
        # Localization img-space rand texturemaps. 
        """,
    config={
        'lr': 0.1,
        # data
        'data_path': './geltip_dataset/dataset/',
        'assets_path': './experimental_setup/geltip/sim_model/assets/',
        'train_dataset': 'sim_geodesic_elastic_bkg',
        '{train_transform}': lambda: transform(image_aug=True, regen_tactile=True),
        'val_dataset': 'sim_geodesic_elastic_bkg',
        '{val_transform}': lambda: transform(image_aug=False, regen_tactile=True),
        '{data_loader}': lambda: loader('train', e['batches_per_epoch'], e['batch_size'], e['train_dataset']),
        '{val_loader}': lambda: loader('val', e['n_val_batches'], e['batch_size'], e['val_dataset']),

        # network
        'model': resnet50(n_activations=2),

        # train
        'loss': nn.MSELoss(),
        '{optimizer}': lambda: optim.Adadelta(e['model'].parameters(), lr=e.lr),
        'epochs': 60,
        'batch_size': 64,
        'batches_per_epoch': 8,
        'feed_size': 32,
        'train_device': 'cuda',

        # validation
        'metrics': [],
        'metrics_names': [],
        'n_val_batches': 4,
        'val_feed_size': 32,
    },
    entry=fit_to_dataset,
    listeners=lambda: [
        Validator(),
        Logger(),
        Plotter(),
        EBoard(),
        ModelSaver()
    ],
    open_e=False,
    src='sim2real'
)
