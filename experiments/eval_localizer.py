from math import sqrt

from dfgiatk.experimenter import run, e

import cv2
import numpy as np

from os import path
from torch import nn, optim
import torch
from dfgiatk.loaders import DatasetSampler
from dfgiatk.loaders.image_loader import LocalizationLabeler, ImageLoader
from dfgiatk.models import resnet50

import imgaug.augmenters as iaa

from dfgiatk.ops.img import cvt_batch, CVT_HWC2CHW
from dfgiatk.train import predict_batch
from sim_model.utils.camera import get_camera_matrix, depth2cloud

split = 'train'
# split = 'val'

# exp, dataset = '2023-02-03 16:07:52', 'real_rgb_aligned' # real2real
# exp, dataset = '2023-02-03 15:15:01', 'sim_plane_realbg' # sim2sim
exp, dataset = '2023-02-06 03:20:22', 'sim_aplane_realbg' # sim2sim large dataset !! change the object locations to adepth

# exp, dataset = '2023-02-03 15:15:01', 'real_rgb_aligned'  # sim2real
# exp, dataset = '2023-02-06 03:20:22', 'real_rgb_aligned'  # sim2real large dataset


w_path = f'outputs/train_localizer/runs/{exp}/out/best_model'

config = {
    'description': """
        Eval localizer
    """,
    'config': {
        # network
        'weights_path': w_path,
        # 'weights_path': '/home/danfergo/Projects/PhD/geltip_simulation/outputs/2023-01-04 12:30:09/out/best_model',
        '{model}': lambda: resnet50(n_activations=2, weights=e.weights_path),

        # train
        'loss': nn.MSELoss(),
        # '{optimizer}': lambda: optim.Adadelta(e['model'].parameters(), lr=e.lr),
        'epochs': 50,
        'batch_size': None,
        'batches_per_epoch': 1,
        'feed_size': 32,
        'train_device': 'cuda',

        # data
        'data_path': './geltip_dataset/dataset/',
        'samples_split': f'depth_{split}_split.yaml',
        'dataset': dataset,
        # 'dataset': 'real_rgb_aligned',
        '{data_loader}': lambda: DatasetSampler(
            samples=DatasetSampler.load_from_yaml(
                path.join(e.data_path, e.samples_split),
                path.join(e.data_path, e.dataset)
            ),
            loader=ImageLoader(
                transform=lambda xs, **kwargs: cvt_batch(iaa.Sequential([
                    iaa.Resize({"height": 120, "width": "keep-aspect-ratio"}),
                ])(images=xs.astype(np.float32) / 255.0), CVT_HWC2CHW).astype(np.float32)
            ),
            labeler=LocalizationLabeler(
                locations_path=path.join(e.data_path, 'adepth_object_locations.yaml')
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

    depth = np.load('./experimental_setup/geltip/sim_assets/bkg.npy')
    cloud_size = (160, 120)
    depth = cv2.resize(depth, cloud_size, interpolation=cv2.INTER_LINEAR)
    cam_matrix = get_camera_matrix(depth.shape[::-1], fov_deg=90)
    cloud = depth2cloud(cam_matrix, depth)

    with torch.inference_mode():
        for batch in iter(data_loader):
            x, y_true = batch
            batch_loss, y_pred = predict_batch(batch)

            x_np = x.cpu().detach().numpy()
            x_np = np.swapaxes(x_np, 1, 2)
            x_np = np.swapaxes(x_np, 2, 3)

            # print('batch size', batch[0].size())

            y_true_np = y_true.detach().cpu().numpy()
            y_pred_np = y_pred.detach().cpu().numpy()

            # # print(.shape)
            # # print(y_pred_np[0][..., 0].shape)
            # # print('---------')Y
            # # print(x.size()[0])
            # print('---------')
            #
            sum_px_err = 0
            sum_m_err = 0
            n_samples = x.size()[0]

            def lim(xx, mn, mx):
                return max(mn, min(xx, mx))

            for i in range(n_samples):
                # print('----------------')
                # print('shapes', y_true_np[i].shape, y_true_np[i].shape)

                y_true_ = tuple([round(c * 0.25) for c in reversed(y_true_np[i].tolist())])
                y_pred_ = tuple([round(c * 0.25) for c in reversed(y_pred_np[i].tolist())])

                pt_true = cloud[y_true_[::-1]]
                pt_pred = cloud[y_pred_[::-1]]

                px_err = sqrt(sum([(y_true_[i] - y_pred_[i]) ** 2 for i in range(2)]))
                m_err = sqrt(sum([(pt_true[i] - pt_pred[i]) ** 2 for i in range(3)]))
                sum_px_err += px_err
                sum_m_err += m_err
                # print('px: ' + str(px_err))
                # print('m: ' + str(m_err * 1000))

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

                # cv2.imshow('gt', np.concatenate([
                #     frame
                #     # y_true_np[i][..., 0],
                #     # y_pred_np[i][..., 0],
                # ], axis=1))
                # cv2.waitKey(-1)
            print('n samples: ', n_samples)
            print('mean px err: ', sum_px_err / n_samples)
            print('mean mm err: ', sum_m_err * 1000 / n_samples)


run(
    **config,
    entry=eval_dataset,
    open_e=False,
    # src='sim2real',
    tmp=True
)
