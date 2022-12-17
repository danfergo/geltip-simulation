from dfgiatk.experimenter import run, e, Logger, Validator, Plotter, EBoard

import yaml

from os import path
from torch import nn, optim

from dfgiatk.loaders import ImageLoader
from dfgiatk.models import resnet50
from dfgiatk.metrics import accuracy

import imgaug.augmenters as iaa

from dfgiatk.train import fit_to_dataset


def loader(partition, epoch_size, batch_size, set):
    return ImageLoader(
        path.join(e.data_path + set),
        yaml.load(open(path.join(e.data_path, partition + '_split.yaml')), yaml.Loader),
        epoch_size=epoch_size,
        batch_size=batch_size,
        transform=e[partition + '_transform']
    )


run(
    description="""
        # ResNet sim2sim lr0.1 sim_geodesic_elastic_bkg (localization). 
        """,
    config={
        'lr': 0.1,
        # data
        'dataset': 'sim_geodesic_elastic_bkg',
        'data_path': './geltip_dataset/dataset/',
        'train_transform': iaa.Sequential([
            # iaa.Multiply(1 / 255.0),
            iaa.Resize({"height": 120, "width": "keep-aspect-ratio"}),
            # iaa.RandAugment(n=2, m=9)
            iaa.OneOf([
                iaa.Affine(rotate=45),
                iaa.AdditiveGaussianNoise(scale=0.2 * 255),
                iaa.Add(50, per_channel=True),
                iaa.Sharpen(alpha=0.5)
            ])
        ]),
        '{data_loader}': lambda: loader('train', e['batches_per_epoch'], e['batch_size'], e['dataset']),
        'val_transform': None,
        '{val_loader}': lambda: loader('val', e['n_val_batches'], e['batch_size'],  e['dataset']),

        # network
        'model': resnet50(n_activations=8),

        # train
        'loss': nn.CrossEntropyLoss(),
        '{optimizer}': lambda: optim.Adadelta(e['model'].parameters(), lr=e.lr),
        'epochs': 50,
        'batch_size': 64,
        'batches_per_epoch': 4,
        'feed_size': 32,
        'train_device': 'cuda',

        # validation
        'metrics': [accuracy],
        'metrics_names': ['Accuracy'],
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
