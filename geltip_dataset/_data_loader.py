import os
from os import listdir
from os.path import isfile, join
import random
import cv2
import numpy as np
import yaml

OBJECT_SET_CLASSES = [
    'wave1',
    'dots',
    'cross_lines',
    'flat_slab',
    'curved_surface',
    'parallel_lines',
    'pacman',
    'torus',
    'cylinder_shell',
    'sphere2',
    'line',
    'cylinder_side',
    'moon',
    'random',
    'prism',
    'dot_in',
    'triangle',
    'sphere',
    'hexagon',
    'cylinder',
    'cone'
]


def to_categorical(y, num_classes=None, dtype='float32'):
    """ retrieved from keras """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def crop_center(img, cropx, cropy):
    y, x = img.shape[:2]
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty + cropy, startx:startx + cropx]


class DataGenerator:

    def __init__(self,
                 dataset_path=None,
                 real_path=None,
                 sim_path=None,
                 depth_path=None,
                 batch_size=32,
                 shuffle=True,
                 output_shape=(224, 224, 3),
                 classes=None,
                 resize=True,
                 split=None,
                 splits_file=None,
                 output_paths=False,
                 augmentor=None):

        dataset_path = dataset_path or ''
        real_path = os.path.join(dataset_path, real_path) if real_path else None
        sim_path = os.path.join(dataset_path, sim_path) if sim_path else None
        depth_path = os.path.join(dataset_path, depth_path) if depth_path else None

        if classes is None:
            classes = OBJECT_SET_CLASSES
        self.c_idx = {c: classes.index(c) for c in classes}
        self.classes = classes
        if split is None:
            files_path = real_path or sim_path or depth_path
            self.samples = [f for f in listdir(files_path) if isfile(join(files_path, f))]
            # and f.split('_')[0] in classes
        else:
            try:
                self.samples = yaml.load(open(splits_file, 'r'))[split]
            except Exception as e:
                print(e)

        # self.samples.sort(key=self.compare_files)

        self.batch_size = len(self.samples) if batch_size is None else batch_size
        self.files_sent = 0
        self.shuffle = shuffle
        self.real_path = real_path
        self.sim_path = sim_path
        self.depth_path = depth_path
        self.resize = resize
        self.output_shape = output_shape
        self.output_paths = output_paths
        self.augmentor = augmentor

    def filename_data(self, file):
        parts = file[:-4].split('_')

        return self.c_idx[parts[0]], int(parts[1])

    def compare_files(self, file):
        cls, k = self.filename_data(file)
        return cls * 1000 + k

    def read_img(self, base, im_path):
        im = cv2.imread(join(base, im_path))
        if self.resize:
            shape = np.shape(im)
            min_side = min(shape[0], shape[1])
            im = crop_center(im, min_side, min_side)
            return cv2.resize(im, (self.output_shape[1], self.output_shape[0]))
        return im

    def read_depth_img(self, base, im_path):
        im = np.load(join(base, im_path[:-4] + '.npy'))
        if self.resize:
            shape = np.shape(im)
            min_side = min(shape[0], shape[1])
            im = crop_center(im, min_side, min_side)
            return cv2.resize(im, (self.output_shape[1], self.output_shape[0]))
        return im

    def size(self):
        return len(self.samples)

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        shuffle = self.shuffle
        samples = self.samples
        batch_size = self.batch_size
        c_idx = self.c_idx
        classes = self.classes
        sim_path = self.sim_path
        depth_path = self.depth_path
        real_path = self.real_path

        if shuffle or self.files_sent < len(samples):
            if shuffle:
                random.shuffle(samples)

            batch_paths = samples[0:batch_size] if shuffle else samples[self.files_sent:self.files_sent + batch_size]

            ret = []
            if real_path is not None:
                ret.append([self.read_img(real_path, x) for x in batch_paths])

            if sim_path is not None:
                ret.append(np.array([self.read_img(sim_path, x) for x in batch_paths]))

            if depth_path is not None:
                ret.append(np.array([self.read_depth_img(depth_path, x) for x in batch_paths]))

            # ys = [y[:-4].split('__') for y in batch_paths]
            # cls = [to_categorical(c_idx[y[0]], len(classes)) for y in ys]
            # ret.append(np.array(cls))

            if self.output_paths:
                ret.append(batch_paths)

            self.files_sent += batch_size

            return self.augmentor(*ret) if (self.augmentor is not None) else tuple(ret)

        raise StopIteration


def load_single_img(path, clones_path=None, ith=0, output_shape=(224, 224, 3), resize=True):
    files = [f for f in listdir(path) if isfile(join(path, f))]
    files.sort()

    def read_img(base, im_path):
        im = cv2.imread(join(base, im_path))
        if resize:
            shape = np.shape(im)
            min_side = min(shape[0], shape[1])
            im = crop_center(im, min_side, min_side)
            return cv2.resize(im, (output_shape[1], output_shape[0]))
        return im

    return read_img(path, files[ith]), read_img(clones_path, files[ith]), files[ith]


def load_single_img2(path, output_shape=(224, 224, 3), resize=True):
    im = cv2.imread(path)
    if resize:
        shape = np.shape(im)
        min_side = min(shape[0], shape[1])
        im = crop_center(im, min_side, min_side)
        return cv2.resize(im, (output_shape[1], output_shape[0]))
    return im


def preview(generator):
    img, cls = next(generator)
    k = 0

    print('Batch Shapes:', np.shape(img), np.shape(cls))
    for i in range(32):
        cv2.imshow('frame', np.concatenate([img[i]], axis=1))
        print(cls[i])
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
        k += 1

# if __name__ == '__main__':
#     r_generator = data_generator(
#         '/home/danfergo/Projects/PhD/gelsight_simulation/dataset/real',
#         # '/home/danfergo/Projects/gelsight_simulation/dataset'
#     )
#     preview(r_generator)
