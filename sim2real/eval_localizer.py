from dfgiatk.experimenter import run, e, Logger, Validator, Plotter, ModelSaver, EBoard

import cv2
import numpy as np

from os import path
from torch import nn, optim
import torch
from dfgiatk.loaders import ImageLoader
from dfgiatk.loaders.image_loader import NumpyMapsLabeler, LocalizationLabeler
from dfgiatk.models import unet, resnet50

import imgaug.augmenters as iaa
from dfgiatk.train import predict_batch


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

            x_np = x.cpu().detach().numpy()
            x_np = np.swapaxes(x_np, 1, 2)
            x_np = np.swapaxes(x_np, 2, 3)

            print('batch size', batch[0].size())

            y_true_np = y_true.detach().cpu().numpy()
            y_pred_np = y_pred.detach().cpu().numpy()

            # # print(.shape)
            # # print(y_pred_np[0][..., 0].shape)
            # # print('---------')Y
            # # print(x.size()[0])
            # print('---------')
            #
            for i in range(x.size()[0]):
                print('----------------')
                print('shapes', y_true_np[i].shape, y_true_np[i].shape)

                y_true_ = tuple([round(c*0.25) for c in reversed(y_true_np[i].tolist())])
                y_pred_ = tuple([round(c*0.25) for c in reversed(y_pred_np[i].tolist())])

                frame = x_np[i]
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.circle(frame,
                                   tuple(y_true_),
                                   10,
                                   (0, 255, 0),
                                   1)
                frame = cv2.circle(frame,
                                   tuple(y_pred_),
                                   10,
                                   (0, 0, 255),
                                   1)

                cv2.imshow('gt', np.concatenate([
                    frame
                    # y_true_np[i][..., 0],
                    # y_pred_np[i][..., 0],
                ], axis=1))
                cv2.waitKey(-1)
            # print('xxx')
def load_model(model, path):
    model.load_state_dict(
        torch.load(path)
    )
    return model

run(
    description="""
        Eval localizer
        """,
    config={
        'lr': 0.1,
        # data
        'data_path': './geltip_dataset/dataset/',
        'samples_split': 'val_split.yaml',
        # 'dataset': 'sim_geodesic_elastic_bkg',
        'dataset': 'real_rgb_aligned',
        '{data_loader}': lambda: ImageLoader(
            samples=ImageLoader.load_from_yaml(
                path.join(e.data_path, e.samples_split),
                path.join(e.data_path, e.dataset)
            ),
            labeler=LocalizationLabeler(
                locations_path=path.join(e.data_path, 'object_locations.yaml')
            ),
            epoch_size=e.batches_per_epoch,
            batch_size=e.batch_size,
            transform=iaa.Sequential([
            #     # iaa.Multiply(1 / 255.0),
                iaa.Resize({"height": 120, "width": "keep-aspect-ratio"}),
            #     # iaa.RandAugment(n=2, m=9)
            #     iaa.OneOf([
            #         # iaa.Affine(rotate=0.1),
            #         iaa.AdditiveGaussianNoise(scale=0.7),
            #         iaa.Add(50, per_channel=True),
            #         iaa.Sharpen(alpha=0.5)
            #     ])
            ])
        ),
        # '{val_loader}': lambda: loader('val', e['n_val_batches'], e['batch_size'], e['val_dataset']),

        # network
        'weights_path': '/home/danfergo/Projects/PhD/geltip_simulation/outputs/2022-12-15 15:26:49/out/best_model',
        '{model}': lambda: load_model(resnet50(n_activations=2), e.weights_path),

        # train
        'loss': nn.MSELoss(),
        '{optimizer}': lambda: optim.Adadelta(e['model'].parameters(), lr=e.lr),
        'epochs': 50,
        'batch_size': None,
        'batches_per_epoch': 1,
        'feed_size': 32,
        'train_device': 'cuda',
    },
    entry=eval_dataset,
    listeners=lambda: [
    ],
    open_e=False,
    src='sim2real'
)
