from dfgiatk.experimenter import run, e, Logger, Validator, Plotter, EBoard

import yaml

from os import path
from torch import nn, optim

from dfgiatk.loaders import ImageLoader
from dfgiatk.loaders.image_loader import LocalizationLabeler, ClassificationLabeler
from dfgiatk.models import resnet50
from dfgiatk.metrics import accuracy

import imgaug.augmenters as iaa

from dfgiatk.train import fit_to_dataset


def loader(partition, epoch_size, batch_size, set):
    samples = ImageLoader.load_from_yaml(
        path.join(e.data_path, partition + '_split.yaml'),
        path.join(e.data_path, set)
    )
    labeler = LocalizationLabeler()

    return ImageLoader(
        samples,
        labeler=labeler,
        epoch_size=epoch_size,
        batch_size=batch_size,
        transform=e[partition + '_transform']
    )


run(
    description="""
        # ResNet sim2sim lr0.1 sim_geodesic_elastic_bkg (localization) (aug) (sim2real). 
        """,
    config={
        'lr': 0.1,
        # data
        'data_path': './geltip_dataset/dataset/',
        'train_dataset': 'sim_geodesic_elastic_bkg',
        'train_transform': iaa.Sequential([
           # iaa.Multiply(1 / 255.0),
           iaa.Resize({"height": 120, "width": "keep-aspect-ratio"}),
           # iaa.RandAugment(n=2, m=9)
           iaa.OneOf([
               iaa.Affine(rotate=0.2),
               iaa.AdditiveGaussianNoise(scale=0.2 * 255),
               iaa.Add(50, per_channel=True),
               iaa.Sharpen(alpha=0.5)
           ])
        ]),
        'val_dataset': 'real_rgb_aligned',
        'val_transform': iaa.Sequential([
            iaa.Resize({"height": 120, "width": "keep-aspect-ratio"}),
        ]),
        '{data_loader}': lambda: loader('train', e['batches_per_epoch'], e['batch_size'], e['train_dataset']),
        '{val_loader}': lambda: loader('val', e['n_val_batches'], e['batch_size'], e['val_dataset']),

        # network
        'model': resnet50(n_activations=2),

        # train
        'loss': nn.MSELoss(),
        '{optimizer}': lambda: optim.Adadelta(e['model'].parameters(), lr=e.lr),
        'epochs': 50,
        'batch_size': 64,
        'batches_per_epoch': 4,
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
        EBoard()
    ],
    open_e=False,
    src='sim2real'
)
