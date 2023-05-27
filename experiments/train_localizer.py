from dfgiatk.experimenter import run, e, Logger, Validator, Plotter, EBoard, ModelSaver

from torch import nn, optim

from dfgiatk.models import resnet50
from dfgiatk.train import fit_to_dataset


from sim2real.shared.loaders import loader
from sim2real.shared.transform import transform

run(
    description="""
        # Localization img-space rand texture-maps. 
        """,
    config={
        'lr': 0.1,

        # data
        'data_path': './geltip_dataset/dataset/',
        'assets_path': './experimental_setup/geltip/sim_model/assets/',
        # 'train_dataset': 'real_rgb_aligned',
        # 'train_dataset': 'sim_plane_realbg',
        'train_dataset': 'sim_aplane_realbg',
        '{train_transform}': lambda: transform(image_aug=True, regen_tactile=True),
        # 'val_dataset': 'real_rgb_aligned',
        # 'val_dataset': 'sim_plane_realbg',
        'val_dataset': 'sim_aplane_realbg',
        '{val_transform}': lambda: transform(image_aug=False, regen_tactile=True),
        '{data_loader}': lambda: loader('train', e['batches_per_epoch'], e['batch_size'], e['train_dataset']),
        '{val_loader}': lambda: loader('val', e['n_val_batches'], e['batch_size'], e['val_dataset']),

        # network
        'model': resnet50(n_activations=2),

        # train
        'loss': nn.MSELoss(),
        '{optimizer}': lambda: optim.Adadelta(e['model'].parameters(), lr=e.lr),
        'epochs': 100,
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
