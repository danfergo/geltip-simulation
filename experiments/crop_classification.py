from os import path

from dfgiatk.experimenter import run, e
from dfgiatk.loaders import DatasetSampler

import torch
import cv2

from dfgiatk.loaders.image_loader import ImageLoader, LocalizationLabeler
from dfgiatk.models import unet, resnet50
from dfgiatk.ops.img import cvt_batch, CVT_HWC2CHW
from dfgiatk.train import predict_batch
import numpy as np

import imgaug.augmenters as iaa



config = {
    'description': """
    Crop classification images
    """,
    'config': {
        # data
        'device': 'cuda',
        # 'feed_size': 4,
        # 'weights_path': '/home/danfergo/Projects/PhD/geltip_simulation/outputs/2023-01-04 10:11:54/out/best_model',
        # '{model}': lambda: resnet50(weights=e.weights_path),
        'data_path': './geltip_dataset/dataset/',
        'samples': 'adepth_all_samples.yaml',
        # 'dataset': 'real_rgb_aligned',
        # 'dataset': 'sim_plane_realbg',
        'dataset': 'sim_aplane_realbg',
        # 'dataset': 'sim_geodesic_elastic_bkg',
        # 'out_dataset': 'real_rgb_cropped',
        # 'out_dataset': 'sim_plane_realbg_cropped',
        'out_dataset': 'sim_aplane_realbg_cropped',
        '{data_loader}': lambda: DatasetSampler(
            samples=DatasetSampler.load_from_yaml(
                path.join(e.data_path, e.samples),
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
            epoch_size=None,
            batch_size=1,
            random_sampling=False,
            return_names=True
        )
    },
}


def prepare_data():
    # data_loader, device, model \
    data_loader, = e[
        'data_loader',
        # 'device',
        # 'model'
    ]
    #
    # model.to(device)

    with torch.inference_mode():
        for x, y_true, names in iter(data_loader):
            # y_pred = predict_batch((x, y_true), compute_loss=False)
            sample_name = names[0]

            y_true_np = y_true.detach().cpu().numpy()
            # y_pred_np = y_pred.detach().cpu().numpy()

            x_np = x.cpu().detach().numpy()
            x_np = np.swapaxes(x_np, 1, 2)
            x_np = np.swapaxes(x_np, 2, 3)

            s = 64
            h, w = 480, 640
            im = x_np[0]  # cv2.imread(sample_name)

            frame = np.zeros((h + s * 2, w + s * 2, 3), dtype=np.uint8)
            frame[s:h + s, s:w + s] = cv2.imread(sample_name)
            im = frame

            y_true_ = tuple([round(c + s) for c in reversed(y_true_np[0].tolist())])
            # y_pred_ = tuple([round(c + s) for c in reversed(y_pred_np[0].tolist())])

            loc = tuple(y_true_)
            # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            # im = cv2.circle(im,
            #                 loc,
            #                 10,
            #                 (0, 255, 0),
            #                 1)
            lx = loc[0]
            ly = loc[1]
            hs = 64
            patch = im[ly - hs:ly + hs, lx - hs:lx + hs]

            s = sample_name
            cv2.imwrite(s.replace(e.dataset, e.out_dataset), patch)
            # cv2.imshow('patch', patch)
            # cv2.waitKey(-1)


run(
    **config,
    src='sim2real',
    entry=prepare_data,
    open_e=False,
    tmp=True
)
