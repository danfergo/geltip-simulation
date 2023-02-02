import os

import cv2
import numpy as np

from sim_model \
    import circle_mask

base = os.path.dirname(os.path.abspath(__file__)) + '/../assets/'

cloud_size = (160, 120)
prefix = str(cloud_size[0]) + 'x' + str(cloud_size[1])
cloud = np.load(base + '/' + prefix + '_ref_cloud.npy')
cloud = cloud.reshape((cloud_size[1], cloud_size[0], 3))

depth = cv2.resize(np.load(base + '/bkg.npy'), cloud_size)
m = circle_mask(cloud_size)


def ij2n(ij):
    return ij[0] * cloud_size[0] + ij[1]


def n2ij(n):
    return n // cloud_size[0], n % cloud_size[0]


def neighbours(n):
    i, j = n2ij(n)
    return [
        *([ij2n([i - 1, j - 1])] if i > 0 and j > 0 else []),
        *([ij2n([i - 1, j])] if i > 0 else []),
        *([ij2n([i - 1, j + 1])] if i > 0 and j < (cloud_size[0] - 1) else []),

        *([ij2n([i, j - 1])] if j > 0 else []),
        *([ij2n([i, j + 1])] if j < (cloud_size[0] - 1) else []),

        *([ij2n([i + 1, j - 1])] if i < (cloud_size[1] - 1) and j > 0 else []),
        *([ij2n([i + 1, j])] if i < (cloud_size[1] - 1) else []),
        *([ij2n([i + 1, j + 1])] if i < (cloud_size[1] - 1) and j < (cloud_size[0] - 1) else []),

    ]


def dist(n1, n2):
    return np.sqrt(np.sum(np.abs(cloud[n2ij(n1)] - cloud[n2ij(n2)]) ** 2))


def shortest_path_map(starting_node):
    n_pixels = cloud_size[0] * cloud_size[1]

    visited = np.zeros(n_pixels)

    previous = np.ones(n_pixels, dtype=np.uint)

    distances = np.ones(n_pixels) * np.inf
    distances[ij2n(starting_node)] = 0

    def assign_best(n, n_):
        new_dist = distances[n] + dist(n, n_)
        if new_dist < distances[n_]:
            distances[n_] = new_dist
            previous[n_] = n

        return distances[n_]

    current_n = ij2n(starting_node)
    i = 0

    while True:
        neighbours_n = neighbours(current_n)
        next_neighbour = np.argmin([assign_best(current_n, n_) if visited[n_] == 0 else np.inf for n_ in neighbours_n])
        visited[current_n] = 1

        current_n = neighbours_n[next_neighbour]

        if visited[current_n] == 1:
            if np.sum(visited) == n_pixels:
                return distances, previous
            current_n = np.argmin([distances[n_] if visited[n_] == 0 else np.inf for n_ in range(len(visited))])

        i = i + 1

        if i % 1000 == 0:
            print(str(round((i / n_pixels) * 100)) + '%')

# def


starting_pt = (60, 10)
starting_n = ij2n(starting_pt)

end_pt = (110, 150)
end_n = ij2n(end_pt)

d, prev = shortest_path_map(starting_pt)

d[d == np.inf] = 0.00000000000001
d = d.reshape(cloud_size[1], cloud_size[0])
d /= np.max(d)

p = end_n
i = 0
while p != starting_n:
    d[n2ij(p)] = 0.0
    print('-->', prev[p], p, n2ij(int(prev[p])), n2ij(p))
    p = int(prev[p])
    i +=1

p = ij2n((10, 150))
i = 0
while p != starting_n:
    d[n2ij(p)] = 0.0
    # print('-->', prev[p], p, n2ij(int(prev[p])), n2ij(p))
    p = int(prev[p])
    i += 1

    cv2.imshow('distances', d)
    cv2.waitKey(-1)

# print(neighbours(ij2n((159, 159))))
# print(ij2n((0, 1)))
# print(ij2n((0, 2)))
# print(ij2n((1, 0)))
# print(ij2n((1, 1)))
# print(ij2n((1, 2)))
