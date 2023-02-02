from dfgiatk.experimenter import run, e, Logger, Validator, Plotter, EBoard, ModelSaver

from os import path
from torch import nn, optim
import numpy as np

from dfgiatk.loaders import DatasetSampler
from dfgiatk.loaders.image_loader import ClassificationLabeler, NumpyMapsLabeler, LocalizationLabeler, ImageLoader
from dfgiatk.models import resnet50
from dfgiatk.metrics import accuracy

from dfgiatk.train import fit_to_dataset


def loader(partition, epoch_size, batch_size, set):
    samples = DatasetSampler.load_from_yaml(
        path.join(e.data_path, partition + '_split.yaml'),
        path.join(e.data_path, set)
    )
    loader = ImageLoader(
        transform=e[partition + '_transform']
    )

    labeler = ClassificationLabeler(samples)

    return DatasetSampler(
        samples,
        loader=loader,
        labeler=labeler,
        epoch_size=epoch_size,
        batch_size=batch_size
    )


def transform(image_aug=False):
    seq = iaa.Sequential(
        [
            iaa.Resize({"height": 120, "width": "keep-aspect-ratio"}),
        ] + ([
                 iaa.OneOf([
                     iaa.Affine(rotate=45),
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


config = {
    'description': """
        # sim2sim-classification-patches. 
    """,
    'config': {
        'lr': 0.1,

        # data
        'data_path': './geltip_dataset/dataset/',
        'train_dataset': 'cropped_classification',
        '{data_loader}': lambda: loader('train', e['batches_per_epoch'], e['batch_size'], e['train_dataset']),
        'train_transform': transform(image_aug=True),
        'val_dataset': 'cropped_classification',
        '{val_loader}': lambda: loader('val', e['n_val_batches'], e['batch_size'], e['val_dataset']),
        'val_transform': transform(image_aug=False),

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
    }
}

run(
    **config,
    entry=fit_to_dataset,
    listeners=lambda: [
        Validator(),
        ModelSaver(),
        Logger(),
        Plotter(),
        EBoard(),
    ],
    open_e=False,
    src='sim2real'
)
