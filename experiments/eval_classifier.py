from math import sqrt

from dfgiatk.experimenter import run, e

import cv2
import numpy as np

from os import path
from torch import nn, optim
import torch
from dfgiatk.loaders import DatasetSampler
from dfgiatk.loaders.image_loader import LocalizationLabeler, ImageLoader, ClassificationLabeler
from dfgiatk.metrics import accuracy
from dfgiatk.models import resnet50

import imgaug.augmenters as iaa

from dfgiatk.ops.img import cvt_batch, CVT_HWC2CHW
from dfgiatk.train import predict_batch

split = 'train'
# split = 'val'
# exp, dataset, s_file = '2023-02-05 14:49:13', 'real_rgb_aligned', ''  # real2real

# exp, dataset, sp = '2023-02-05 21:33:24', 'sim_plane_realbg', ''  # sim2sim
# exp, dataset, sp = '2023-02-05 17:09:27', 'sim_aplane_realbg', ''  # sim2sim extended

# exp, dataset, sp = '2023-02-05 21:33:24', 'real_rgb_aligned', ''  # sim2real
# exp, dataset, sp = '2023-02-05 17:09:27', 'real_rgb_aligned', ''  # sim2real


# -- cropped stuff
# exp, dataset, sp = '2023-02-05 18:39:53', 'real_rgb_cropped', ''  # real2real cropped
# exp, dataset, sp = '2023-02-05 22:38:09', 'sim_plane_realbg_cropped', ''  # sim2real
# exp, dataset, sp = '2023-02-06 02:35:50', 'sim_aplane_realbg_cropped', ''  # sim2real large dataset

# exp, dataset, sp = '2023-02-05 22:38:09', 'real_rgb_cropped', ''  # sim2real cropped
exp, dataset, sp = '2023-02-06 02:35:50', 'real_rgb_cropped', ''  # sim2real large dataset cropped


w_path = f'outputs/train_classifier/runs/{exp}/out/best_model'

config = {
    'description': """
        Eval localizer
    """,
    'config': {
        # network
        'weights_path': w_path,
        '{model}': lambda: resnet50(n_activations=8, weights=e.weights_path),

        # train
        'loss': nn.CrossEntropyLoss(),
        'epochs': 50,
        'batch_size': None,
        'batches_per_epoch': 1,
        'feed_size': 32,
        'train_device': 'cuda',

        # data
        'data_path': './geltip_dataset/dataset/',
        'samples_split': f'{sp}depth_{split}_split.yaml',
        'dataset': dataset,
        # 'dataset': 'real_rgb_aligned',
        '{samples}': lambda: DatasetSampler.load_from_yaml(
            path.join(e.data_path, e.samples_split),
            path.join(e.data_path, e.dataset)
        ),
        '{data_loader}': lambda: DatasetSampler(
            samples=e.samples,
            loader=ImageLoader(
                transform=lambda xs, **kwargs: cvt_batch(iaa.Sequential([
                    iaa.Resize({"height": 120, "width": "keep-aspect-ratio"}),
                ])(images=xs.astype(np.float32) / 255.0), CVT_HWC2CHW).astype(np.float32)
            ),
            labeler=ClassificationLabeler(
                samples=e.samples
            ),
            epoch_size=e.batches_per_epoch,
            batch_size=None,
            return_names=False,
            random_sampling=False,
            device='cuda'
        ),

    }
}


def eval_dataset():
    epochs, batch_size, batches_per_epoch, data_loader, train_device, model = e[
        'epochs',
        'batch_size',
        'batches_per_epoch',
        'data_loader',
        'train_device',
        'model'
    ]

    model.to(train_device)

    with torch.inference_mode():
        for batch in iter(data_loader):
            x, y_true = batch
            batch_loss, y_pred = predict_batch(batch)

            acc = accuracy(y_pred, y_true).item()
            print('accuracy', acc)


run(
    **config,
    entry=eval_dataset,
    open_e=False,
    # src='sim2real'
)
