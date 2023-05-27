from dfgiatk.experimenter import run, e, Logger, Validator, Plotter, EBoard, ModelSaver

from torch import nn, optim
from dfgiatk.models import resnet50
from dfgiatk.metrics import accuracy

from dfgiatk.train import fit_to_dataset

from sim2real.shared.loaders import loader
from sim2real.shared.transform import transform

config = {
    'description': """
        # sim2sim-classification-patches. 
    """,
    'config': {
        'lr': 0.1,

        # data
        'data_path': './geltip_dataset/dataset/',
        # 'train_dataset': 'real_rgb_cropped',
        # 'train_dataset': 'sim_aplane_realbg',
        # 'train_dataset': 'cropped_classification',
        # 'train_dataset': 'sim_plane_realbg',
        # 'train_dataset': 'sim_plane_realbg_cropped',
        'train_dataset': 'sim_aplane_realbg_cropped',
        '{data_loader}': lambda: loader('train', e['batches_per_epoch'], e['batch_size'], e['train_dataset'], labels='classes'),
        'train_transform': transform(image_aug=True),
        # 'val_dataset': 'real_rgb_cropped',
        # 'val_dataset': 'real_rgb_aligned',
        # 'val_dataset': 'sim_aplane_realbg',
        # 'val_dataset': 'cropped_classification',
        # 'val_dataset': 'sim_plane_realbg',
        # 'val_dataset': 'sim_plane_realbg_cropped',
        'val_dataset': 'sim_aplane_realbg_cropped',
        '{val_loader}': lambda: loader('val', e['n_val_batches'], e['batch_size'], e['val_dataset'], labels='classes'),
        'val_transform': transform(image_aug=False),

        # network
        'model': resnet50(n_activations=8),

        # train
        'loss': nn.CrossEntropyLoss(),
        '{optimizer}': lambda: optim.Adadelta(e['model'].parameters(), lr=e.lr),
        'epochs': 100,
        'batch_size': 64,
        'batches_per_epoch': 8,
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
