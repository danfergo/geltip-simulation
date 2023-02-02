from os import path

from dfgiatk.experimenter import e
from dfgiatk.loaders import DatasetSampler
from dfgiatk.loaders.image_loader import NumpyMapsLabeler, LocalizationLabeler, ImageLoader


def localizer_loader(partition, epoch_size, batch_size, set, load_depth=False):
    samples = DatasetSampler.load_from_yaml(
        path.join(e.data_path, partition + '_split.yaml'),
        path.join(e.data_path, set)
    )
    loader = NumpyMapsLabeler(
        path.join(e.data_path, 'sim_depth'),  # _aligned
        transform=e[partition + '_transform']
    ) if load_depth else ImageLoader(
        transform=e[partition + '_transform']
    )

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


