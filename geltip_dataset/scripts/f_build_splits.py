from os import path
import random
import math
import yaml


def split(samples, train_partition=0.8):
    i = math.ceil(len(samples) * train_partition)
    train = samples[: i]
    val = samples[i:]
    return train, val


def main():
    dataset_path = '/home/danfergo/Projects/PhD/geltip_simulation/geltip_dataset/dataset'
    objects = ['cone', 'sphere', 'random', 'cylinder', 'cylinder_shell', 'pacman', 'dot_in', 'dots']
    N_ROWS = 3
    N_CONTACTS = 6
    train_partition = 0.8

    samples = [path.join(o, f"{i}_{j}.png") for j in range(N_CONTACTS) for i in range(N_ROWS) for o in objects]
    random.shuffle(samples)

    train, val = split(samples, train_partition)

    yaml.dump(train, open(path.join(dataset_path, 'train_split.yaml'), 'w'))
    yaml.dump(val, open(path.join(dataset_path, 'val_split.yaml'), 'w'))


if __name__ == '__main__':
    main()

    # for obj in objects:
    #
    #     for i in range(N_ROWS):
    #         for j in range(N_CONTACTS):
