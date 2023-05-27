from os import path

from dfgiatk.experimenter import e
from dfgiatk.loaders import DatasetSampler
from dfgiatk.loaders.image_loader import NumpyMapsLabeler, LocalizationLabeler, ImageLoader, ClassificationLabeler


def loader(partition, epoch_size, batch_size, set, labels='locations', load_depth=False):
    is_extended_set = set.split('_')[1].startswith('a')

    splits_file = f'adepth_{partition}_split.yaml' if is_extended_set else f'depth_{partition}_split.yaml'

    samples = DatasetSampler.load_from_yaml(
        path.join(e.data_path, splits_file),
        path.join(e.data_path, set)
    )

    trans = e[partition + '_transform']
    depth_samples = path.join(e.data_path, 'sim_depth')
    loader_ = NumpyMapsLabeler(depth_samples, transform=trans) \
        if load_depth \
        else ImageLoader(transform=trans)

    locations_file = 'adepth_object_locations.yaml' \
        if is_extended_set \
        else 'object_locations.yaml'

    labeler = LocalizationLabeler(locations_path=path.join(e.data_path, locations_file)) \
        if labels == 'locations' \
        else ClassificationLabeler(samples)

    return DatasetSampler(
        samples,
        loader=loader_,
        labeler=labeler,
        epoch_size=epoch_size,
        batch_size=batch_size
    )
